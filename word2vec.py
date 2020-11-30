"""
Implementation of the Word2Vec model architecture with subsampling and negative sampling.
"""

from typing import List
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def load_corpus(filenames: List[Path], compression_type: str = None,
                max_vocab_size: int = None, sequence_length: int = None) -> tf.data.Dataset:
    """
    Load a corpus from the given files as tf.data.Dataset object.

    Args:
        filenames: A list of filenames to load from.
        compression_type: One of "" (no compression), "ZLIB", or "GZIP".
        max_vocab_size: The maximum number of tokens in the vocabulary.
            If not specified, there is no limit on the number of tokens.
        sequence_length: The length of each line after tokenization and
            vectorization. Pads or truncates samples to the same length.
    """
    def _remove_empty(x: tf.Tensor) -> tf.Tensor:
        """Filter out empty strings from the given string tensor."""
        # Convert the string tensor to a boolean tensor (where each string element
        # is replaced with True if it is not empty, or False otherwise).
        return tf.cast(tf.strings.length(x), bool)

    # Load the files using a TextLineDataset object, which will automatically
    # decompress the files (if needed) and split them into lines.
    dataset = tf.data.TextLineDataset(
        filenames=filenames,
        compression_type=compression_type
    ).filter(_remove_empty)

    # The TextVectorization layer will create a vocabulary from a string tensor.
    # In a nut shell, it standardises the data (lower and strip punctuation),
    # tokenizes it, and then assigns an integer index to each word.
    vectorizer = TextVectorization(
        max_tokens=max_vocab_size,
        output_sequence_length=sequence_length
    )

    # Apply our data onto the vectorizer. Now, we can pass strings into the vectorizer
    # and it will give us vectors for use in our model.
    vectorizer.adapt(dataset.batch(1024))

    # A flag that tells TensorFlow to tune dataset pipeline parameters automatically.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # Vectorize our data
    dataset.batch(1024).prefetch(AUTOTUNE).map(vectorizer).unbatch()
    return dataset


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'extra-imports': [
            'typing',
            'pathlib',
            'tensorflow',
            'tensorflow.keras.layers.experimental.preprocessing'
        ],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
