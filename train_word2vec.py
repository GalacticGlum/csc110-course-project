"""Train a Word2Vec model on a corpus."""

import time
import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
from logger import logger
from utils import set_seed, get_next_run_id
from word2vec import (
    Tokenizer,
    Word2Vec,
    make_dataset
)


def main(args: argparse.Namespace) -> None:
    """Main entrypoint for the script."""
    # Make sure we have at least one file.
    if len(args.filenames) == 0:
        logger.error('At least one text file is required!')
        exit(1)

    set_seed(args.seed)
    tokenizer = Tokenizer(
        max_tokens=args.max_tokens,
        min_word_frequency=args.min_word_frequency,
        sample_threshold=args.sample_threshold
    )

    start_time = time.time()
    logger.info('Building vocabulary from corpus...')

    tokenizer.build(filenames=args.filenames)

    logger.info('Finished building vocabulary (took {:.2f} seconds)'.format(
        time.time() - start_time
    ))

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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or args.filenames[0].stem
    logdir = args.output_dir / get_next_run_id(args.output_dir, run_name)

    logger.info(f'Starting training (for {args.epochs} epochs).')
    model.train(
        dataset, logdir, args.initial_lr, args.target_lr,
        args.log_freq, args.save_freq
    )

    # Save embeddings and vocab
    #
    # The weights of the projection layer are components of the
    # embedding vectors. The i-th row of the weight matrix is the
    # embedding vector for the word whose encoded index is i.
    proj = model.weights[0].numpy()
    np.save(logdir / 'proj_weights', proj)
    # Save the tokenizer state
    tokenizer.save(logdir / 'tokenizer.json')
    # Save a list of the vocabulary words
    with open(logdir / 'vocab.txt', 'w') as file:
        for word in tokenizer.words:
            file.write(f'{word}\n')


if __name__ == '__main__':
    # import python_ta
    # python_ta.check_all(config={
    #     'extra-imports': [
    #         'argparse',,
    #         'pathlib',
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
    parser.add_argument('-o', '--output-dir', default='./output/word2vec/', type=Path,
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
    parser.add_argument('--lambda', dest='lambda_power', type=float, default=0.75,
                        help='Used to skew the probability distribution when sampling words.')
    parser.add_argument('--no-bias', action='store_false', dest='bias',
                        help='Don\'t add a bias term.')
    # Training configuration
    parser.add_argument('--run-name', type=str, default=None,
                        help='The name of the run. If unspecified, defaults to the name of the '
                        'first file in the given corpus.')
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
                        help='The frequency at which to log stats, '
                             'in global steps. Defaults to 1000.')
    parser.add_argument('--save-freq', type=int, default=10000,
                        help='The frequency at which to save the model, '
                             'in global steps. Defaults to 10000.')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Sets the seed of the random engine. '
                              'If unspecified, a random seed is chosen.')
    main(parser.parse_args())