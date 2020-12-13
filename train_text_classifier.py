"""Train a text multi-class text classifier using a pretrained BERT
(bi-directional encoder representations from transformers) model checkpoint.

This script takes in a dataset of (feature, label) pairs where feature is a string
and label is an integer denoting the class of the input example.
"""
import argparse
from pathlib import Path
from typing import Tuple, Union, Optional

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

from logger import logger
from utils import set_seed, get_next_run_id


def load_dataset(directory: Union[str, Path], validation_split: Optional[float] = 0.2,
                 batch_size: Optional[int] = 32, train_folder_name: Optional[str] = 'train',
                 test_folder_name: Optional[str] = 'test', seed: Optional[int] = 0) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int]:
    """Load the given dataset into training, validation, and testing sets.
    Return the training, validation, and testing datasets, and the number of
    classes as a tuple.

    Args:
        directory: Path to the root dataset directory. The dataset should already be split into
            a training and testing set. The training and testing sets should consists of folders
            for each distinct class of the classifier, where each folder contains a set of text
            files (examples for that class).

            For example, if there are two classes: positive and negative. The dataset will contain
            two folders with names "training" and "testing". And then, each of these folders will
            contain two more folders named "positive" and "negative", where each folder has one or
            more text files with positive and negative examples respectively.
        validation_split: The proportion of the training set used for validation.
        train_folder_name: The name of the folder containing the training set.
        test_folder_name: The name of the folder containing the testing set.
        seed: The seed to use when splitting and shuffling the data. This ensures that there is no
            overlap between the training and validation subsets.

    Preconditions:
        - 0 <= validation <= 1
    """
    def _optimize_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Return a dataset with caching and prefetching enabled."""
        return dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Convert directory to pathlib object
    directory = Path(directory)
    # Load the validation dataset
    train_dataset = tf.keras.preprocessing.text_dataset_from_directory(
        str(directory / train_folder_name),
        batch_size=batch_size,
        validation_split=validation_split,
        subset='training',
        seed=seed
    )

    num_classes = len(train_dataset.class_names)
    train_dataset = _optimize_dataset(train_dataset)

    # Load the validation dataset
    validation_dataset = tf.keras.preprocessing.text_dataset_from_directory(
        str(directory / train_folder_name),
        batch_size=batch_size,
        validation_split=validation_split,
        subset='validation',
        seed=seed
    )

    validation_dataset = _optimize_dataset(validation_dataset)

    # Load the testing dataset
    test_dataset = tf.keras.preprocessing.text_dataset_from_directory(
        str(directory / test_folder_name),
        batch_size=batch_size
    )

    test_dataset = _optimize_dataset(test_dataset)
    return train_dataset, validation_dataset, test_dataset, num_classes


DEFAULT_BERT_PREPROCESS_HANDLE = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1'
DEFAULT_BERT_MODEL_HANDLE = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'


def build_classifier_model(num_classes: int, dropout_rate: Optional[float] = 0.1,
                           bert_preprocess_handle: Optional[str] = DEFAULT_BERT_PREPROCESS_HANDLE,
                           bert_model_handle: Optional[str] = DEFAULT_BERT_MODEL_HANDLE) \
        -> tf.keras.Model:
    """Build the text classification model with a BERT backbone and a text classification head.
    The network takes the outputs of the BERT model and passes it through a single fullly
    connected/linear layer to predict the multi-class probability distribution.

    Args:
        num_classes: The number of classes to predict (i.e. units in the output layer).
        dropout_rate: The probability of dropping a unit for a single training step.
        bert_preprocess_handle: The TF Hub handle for the pretrained preprocessing model.
        bert_model_handle: The TF Hub handle for the pretrained BERT model.
            Defaults to a small BERT model with 4 stacked encoders, and a
            hidden layer size of 512 units (i.e. L-4_H-512).
    """

    x = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(bert_preprocess_handle, name='preprocessing')
    encoded_x = preprocessing_layer(x)
    encoder = hub.KerasLayer(bert_model_handle, trainable=True, name='BERT_encoder')
    outputs = encoder(encoded_x)
    classifier = outputs['pooled_outputs']
    # Apply dropout
    classifier = tf.keras.layers.Dropout(dropout_rate)(classifier)
    classifier = tf.keras.layers.Dense(num_classes, activation=None, name='classifier')(classifier)
    return tf.keras.Model(x, classifier)


def main(args: argparse.Namespace) -> None:
    """Main entrypoint for the script."""
    set_seed(args.seed)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or args.output_dir.stem
    logdir = args.output_dir / get_next_run_id(args.output_dir, run_name)

    # Load datasets
    train_dataset, validation_dataset, test_dataset, num_classes = load_dataset(
        args.dataset_directory,
        validation_split=args.validation_split,
        batch_size=args.batch_size,
        train_folder_name=args.train_folder_name,
        test_folder_name=args.test_folder_name
    )

    # Build the model
    model = build_classifier_model(
        num_classes,
        dropout_rate=args.dropout_rate,
        bert_preprocess_handle=args.bert_preprocess_handle,
        bert_model_handle=args.bert_model_handle
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = tf.metrics.SparseCategoricalAccuracy()

    steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    total_train_steps = steps_per_epoch * args.epochs
    warmup_steps = int(0.1 * total_train_steps)

    # Load the optimizer
    optimizer = optimization.create_optimizer(
        init_lr=args.initial_lr,
        num_train_steps=total_train_steps,
        num_warmup_steps=warmup_steps,
        optimizer_type='adamw'
    )

    # Compile the model with the optimizer, loss, and metrics
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Create training callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=logdir / 'model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss'
    )

    # Train the model
    logger.info('Starting training with {} (for {} epochs).'.format(
        args.bert_model_handle, args.epochs
    ))

    model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=[
            tensorboard_callback,
            checkpoint_callback
        ]
    )

    # Evaluate the model
    logger.info('Evaluating the model on the testing dataset')
    loss, accuracy = model.evaluate(test_dataset)
    logger.info('Loss: {}'.format(loss))
    logger.info('Accuracy: {}'.format(accuracy))

    # Save final model
    model.save(logdir / 'model_final.hd5')


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'extra-imports': [
            'argparse',
            'pathlib',
            'typing',
            'tensorflow',
            'tensorflow_hub',
            'tensorflow_text',
            'official.nlp',
            'logger',
            'utils',
        ],
        'allowed-io': ['main'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    parser = argparse.ArgumentParser(description='Train a multi-class text classifier '
                                     'using a pretrained BERT model from TF Hubs.')
    parser.add_argument('dataset_directory', type=Path, help='Path to the root dataset directory. '
                        'The dataset should already be split into a training and testing set.')
    parser.add_argument('-o', '--output-dir', default='./output/classifier/', type=Path,
                        help='Output directory to save the trained model.')
    # Model configuration
    parser.add_argument('--bert-preprocess-handle', type=str, default=DEFAULT_BERT_PREPROCESS_HANDLE,
                        help='The TF Hub handle for the pretrained preprocessing model.')
    parser.add_argument('--bert-model-handle', type=str, default=DEFAULT_BERT_MODEL_HANDLE,
                        help='The TF Hub handle for the pretrained BERT model.')
    parser.add_argument('--dropout-rate', type=float, default=0.1,
                        help='The probability of dropping a unit for a single training step.')
    # Dataset configuration
    parser.add_argument('--train-folder-name', type=str, default='train',
                        help='The name of the folder containing the training set.')
    parser.add_argument('--test-folder-name', type=str, default='test',
                        help='The name of the folder containing the testing set.')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='The proportion of the training set used for validation.')
    # Training configuration
    parser.add_argument('--run-name', type=str, default=None,
                        help='The name of the run. If unspecified, defaults to the '
                        'name of the directory')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='The size of a single training batch. Defaults to 32.')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='The number of times to iterate over the dataset '
                             'while training. Defaults to 10.')
    parser.add_argument('--initial-lr', type=float, default=3e-5,
                        help='The initial learning rate.')
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