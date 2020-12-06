"""Train a Word2Vec model on a corpus."""

import datetime
import argparse
import tensorflow as tf
from pathlib import Path
from typing import Optional, List

from tqdm import tqdm
from logger import logger
from utils import set_seed
from word2vec import (
    Tokenizer,
    Word2Vec,
    make_dataset
)


def main(args: argparse.Namespace) -> None:
    """Main entrypoint for the script."""
    set_seed(args.seed)

    tokenizer = Tokenizer()
    logger.info('Building vocabulary from corpus...')
    tokenizer.build(filenames=args.filenames)

    dataset = make_dataset(args.filenames, tokenizer,
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    model = Word2Vec(tokenizer,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        n_negative_samples=args.n_negative_samples,
        lambda_power=args.lambda_power,
        bias=args.bias
    )

    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Create summary writers (for logging to tensorboard)
    summary_writer = tf.summary.create_file_writer(args.outdir)

    # Use stochastic gradient descent without learning rate
    # We will apply the learning rate ourselves, with a custom decay.
    optimizer = tf.keras.optimizers.SGD(1.0)
    logger.info(f'Starting training (for {args.epochs} epochs).')

    @tf.function(input_signature=tf.TensorSpec(shape=(args.batch_size,), dtype=tf.int64))
    def _train_step(inputs: tf.Tensor, labels: tf.Tensor, iter_progress: tf.Tensor,
                model: Word2Vec, optimizer: tf.keras.optimizers.Optimizer) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Train the word2vec model for a single step.

        This function computes the training loss and performs backprop using an optimizer.

        Args:
            inputs: Input features to the model (targets).
            labels: Correct labels for the model (contexts).
            model: The model to train.
            optimizer: An optimizer instance.
        """
        loss = model(inputs, labels)
        # Compute gradients
        gradients = tf.gradients(loss, model.trainable_variables)

        # Linearly interpolate between the initial and target learning rate.
        t = iter_progress[0]
        learning_rate = args.initial_lr * (1 - t) + args.target_lr * t
        learning_rate = tf.maximum(learning_rate, args.target_lr)

        # Apply learning rate to the gradients for each trainable variable/weight.
        # In our case, this is for the proj, proj_out, and bias layer.
        for i in range(len(model.trainable_variables)):
            if hasattr(gradients[i], '_values'):
                gradients[i]._values *= learning_rate
            else:
                gradients[i] *= learning_rate

        # Apply gradients for backprop
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, learning_rate

    average_loss = 0
    with tqdm(dataset) as progress_bar:
        for global_step, (inputs, labels, iter_progress) in tqdm(dataset):
            loss, learning_rate = _train_step(
                inputs, labels, progress,
                model, optimizer
            )

            average_loss += loss.numpy().mean()
            if global_step % args.log_freq:
                if global_step > 0:
                    average_loss /= args.log_freq

                # Output summary information
                progress_bar.write(
                    f'global step: {global_step}, '
                    f'average loss: {average_loss}, '
                    f'learning rate: {learning_rate}'
                )

                with summary_writer.as_default():
                    tf.summary.scalar('lr', learning_rate, step=global_step)
                    tf.summary.scalar('loss', average_loss, step=global_step)

                # Reset average loss
                average_loss = 0

    # Output embeddings
    proj = model.weights[0].numpy()
    np.save(args.outdir / 'proj_weights', proj)
    # Save the tokenizer state
    tokenizer.save(args.outdir / 'vocab.json')
    # Save a list of the vocabulary words
    with tf.io.gfile.GFile(args.outdir / 'vocab.txt', 'w') as file:
        for word in tokenizer.words:
            file.write(f'{word}\n')


if __name__ == '__main__':
    # import python_ta
    # python_ta.check_all(config={
    #     'extra-imports': [
    #         'argparse',,
    #         'pathlib',
    #         'tqdm',
    #         'utils',
    #         'logger',
    #         'word2vec'
    #     ],
    #     'allowed-io': ['main'],
    #     'max-line-length': 100,
    #     'disable': ['R1705', 'C0200']
    # })

    parser = argparse.ArgumentParser(description='Train a Word2Vec model on a corpus.')
    parser.add_argument('filenames', type=Path, nargs='+',
                        help='Names of text files to train on (the corpus).')
    parser.add_argument('-o', '--outdir', type=Path,
                        help='Output directory to save model weights, embeddings, and vocab.')

    # Tokenizer configuration
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='The maximum number of tokens in the vocabulary. '
                              'If unspecified, there is no max.')
    parser.add_argument('--min-word-frequency', type=int, default=0,
                        help='The minimum frequency for words to be included '
                             'in the vocabulary. Default to 0.')
    parser.add_argument('-t', '--sample-threshold', type=float, default=1e-3,
                        help='A small value to offset the probabilities for '
                             'sampling any given word. This is the "t" variable '
                             'in the distribution given by Mikolov et. al. in '
                             'their Word2Vec paper. Defaults to 0.001.')
    # Model configuration
    parser.add_argument('-w', '--window-size', type=int, default=5,
                        help='The size of the sliding window. Defaults to 5.')
    parser.add_argument('-ns', '--n-negative-samples', type=int, default=5,
                        help='Number of negative samples to construct per positive sample. '
                        'Defaults to 5, meaning that 6N training examples are created, '
                        'where N is the number of positive samples.')
    parser.add_argument('-hs', '--hidden-size', type=int, default=256,
                        help='The number of units in the hidden layers. '
                        'The dimensionality of the embedding vectors. Defaults to 256.')
    parser.add_argument('--lambda', type=float, default=0.75,
                        help='Used to skew the probability distribution when sampling words.')
    parser.add_argument('--no-bias', action='store_false', dest='bias',
                        help='Don\'t add a bias term.')
    # Training configuration
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='The size of a single training batch. Defaults to 256.')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='The number of times to iterate over the dataset '
                              'while training. Defaults to 1.')
    parser.add_argument('--initial-lr', type=float, default=0.025,
                        help='The initial learning rate.')
    parser.add_argument('--target-lr', type=float, default=1e-4,
                        help='The target learning rate.')
    parser.add_argument('--log-freq', type=int, default=1000,
                        help='The frequency at which to save the model, '
                             'in global steps. Defaults to 1000.')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Sets the seed of the random engine. '
                              'If unspecified, a random seed is chosen.')
    main(parser.parse_args())