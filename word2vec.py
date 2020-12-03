"""Implementation of the Word2Vec model architecture with subsampling and negative sampling."""

import time
import random
from pathlib import Path
from typing import (
    Tuple,
    List,
    Dict,
    Optional,
    Union
)

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from logger import logger
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def make_skipgram_pairs(sequence: List[int], window_size: int,
                        sampling_table: Optional[np.ndarray] = None,
                        skip_zero: Optional[bool] = True,
                        skip_func: Optional[callable] = None) \
        -> List[Tuple[int, int]]:
    """
    Make positive example pairs for a skip-gram language model.

    Return a list of 2-element tuples consisting of the target and
    context word (in the neighbourhood of the given window size).

    Args:
        sequence: A sequence of words encoded as a list of integers.
        window_size: The size of the sliding window (on each side of the
            target word) used to construct sample pairs.

            For every index i in the sequence, contexts words are chosen in
            the neigbourhood defined by [i - window_size, i + window_size].
        sampling_table: The probability of sampling words by frequency.
            The i-th element of the table gives the probability of sampling
            the word with (encoded) index, i, in the vocabulary.

            For example, if "the" is encoded as 7, then sampling_table[7]
            should give the sampling probability for that word ("the").
        skip_zero: Whether to skip words with index 0 in the sequence. By convention,
            we assume that the vocabulary is one-indexed, and that 0 is not a valid
            word index. If False, 0 is treated as a valid word index.
        skip_func: A filtering function which takes in a word as input and returns a
            boolean indicating whether it should be skipped.

    Preconditions:
        - window_size >= 0

    >>> sequence = [4, 3, 5, 5, 2, 0, 0]
    >>> expected = [(4, 3), (4, 5),\
                    (3, 4), (3, 5), (3, 5),\
                    (5, 4), (5, 3), (5, 5), (5, 2),\
                    (5, 3), (5, 5), (5, 2),\
                    (2, 5), (2, 5)]
    >>> expected == make_skipgram_pairs(sequence, 2)
    True
    >>> make_skipgram_pairs([1, 2], 0)
    []
    """
    def _skip_word(x):
        """Return whether to skip the word."""
        return skip_zero and not x or \
               skip_func is not None and skip_func(x)

    n = len(sequence)
    pairs = []
    for i, target_word in enumerate(sequence):
        if _skip_word(target_word):
            continue

        if sampling_table is not None:
            p = sampling_table[target_word]
            # Sample iff random.random() < p.
            if random.random() >= p:
                continue

        start_index = max(0, i - window_size)
        end_index = min(n, i + window_size + 1)
        # Iterate through the neighbourhood the target word.
        for j in range(start_index, end_index):
            # We don't want the words to ever be the same!
            if i == j:
                continue

            # Create new (target, context) pair.
            context_word = sequence[j]
            if _skip_word(context_word):
                continue

            pairs.append((target_word, context_word))

    return pairs


def make_training_data(sequences: List[List[int]], window_size: int,
                       n_negative_samples: int, vocab_size: int,
                       use_subsampling: Optional[bool] = True,
                       show_progress_bar: Optional[bool] = True) \
        -> Tuple[List[int], List[int], List[int]]:
    """Make training data for a skip-gram language model.

    Return a 3-element tuple consisting of the target words, context words,
    and labels.

    Args:
        sequences: A list of vectorised word token sequences.
        window_size: The size of the sliding window used to construct sample pairs.
        n_negative_samples: The number of negative samples to generate per
            positive context word. Mikolov, et. al. showed that for small
            datasets, values between 5 and 20 (inclusive) work best, whereas
            for large datasets, values between 2 and 5 (inclusive) suffice.
        vocab_size: The nubmer of words (tokens) in the model vocabulary.
        use_subsampling: Whether to subsample words based on frequency probabilities.
            If False, words are uniformly sampled from the vocabulary.
        show_progress_bar: Whether to show a progress bar while the jobs run.

    Preconditions:
        - subsampling_table is None or len(subsampling_table) == vocab_size
    """
    targets, contexts, labels = [], [], []
    # TODO:The make_sampling_table functions gives an array where the i-th
    # element gives the probabiltiy of sampling the i-th most common word under
    # the assumption that the word frequency follows a Zipf-like distribution.
    #
    # For natural text, this is often a good enough approximation for the true
    # word frequency; however, we can do better by actually computing the frequency
    # of each word in the vocabulary.
    if use_subsampling:
        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    else:
        sampling_table = None

    for sequence in tqdm(sequences, disable=not show_progress_bar):
        skipgram_pairs = make_skipgram_pairs(
            sequence,
            window_size,
            sampling_table=sampling_table
        )

        # Create n_negative_samples for each positve (target, context) word-pair,
        # by sampling random words from the vocabulary (excluding the context word).
        #
        # A negative sample refers to a (target, not_context) word-pair where not_context
        # is NOT the context word of the positive sample (i.e. not_context != context).
        #
        # We sample words from the vocabulary according to a Zipfian distribution.
        # TODO: The intuition towards negative sampling, as proposed by Mikolov et. al.
        # is to sample words from the vocabulary from a distribution designed to favour
        # more frequent words. The probability of sampling a word i is given by:
        #   P(w_i) = [f(w_i)^lambda] / sum(f(w_j)^lambda for j = 0 to n),
        # where n is the vocabulary size, f : N -> N, gives the frequency of each word
        # in the vocabulary, and lambda is a hyperparameter (set to 3/4 in the paper).
        #
        # As mentioned above, we can approximate the frequency using a Zipf-like distribution
        # for natural text. But, we can't beat actually knowing the frequency of each word.
        for target_word, context_word in skipgram_pairs:
            # Create a tensor of shape (1, 1) that contains the context word.
            # This is the shape expected by log_uniform_candidate_sampler.
            true_classes = tf.expand_dims(tf.constant([context_word], dtype=tf.int64), 1)
            # Sample a random word (index) from the range [0, vocab_size) excluding the
            # elements of the true_classes tensor (in our case, the context word).
            #
            # Returns an int tensor with shape (n_negative_samples,) containing the sampled values.
            negative_samples, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=true_classes,
                num_true=1,
                num_sampled=n_negative_samples,
                unique=True,  # Sample without replacement
                range_max=vocab_size,
            )

            # Expand dimension of negaitve samples to match true_classes
            # We need to do this for concatenation to work (outer dims need to match).
            negative_samples = tf.expand_dims(negative_samples, 1)
            # Concatenate the negative samples to the positive sample
            context = tf.concat([true_classes, negative_samples], 0)
            # label is a int tensor of the form 1, 0, ..., 0, where the element at index i
            # indicates whether context[i] is a positive (1) or negative (0) sample.
            label = tf.constant([1] + [0] * n_negative_samples, dtype=tf.int64)

            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


def load_dataset(filenames: List[Union[Path, str]],
                 window_size: int, n_negative_samples: int,
                 compression_type: Optional[str] = None,
                 max_vocab_size: Optional[int] = None,
                 sequence_length: Optional[int] = None,
                 batch_size: Optional[int] = 1024,
                 shuffle_buffer_size: Optional[int] = 10000,
                 vectorization_batch_size: Optional[int] = 1024) \
        -> tf.data.Dataset:
    """
    Load a corpus from the given files as a tf.data.Dataset object.

    Args:
        filenames: A list of filenames to load from.
        window_size: The size of the sliding window used to construct sample pairs.
        n_negative_samples: The number of negative samples to generate per
            positive context word. Mikolov, et. al. showed that for small
            datasets, values between 5 and 20 (inclusive) work best, whereas
            for large datasets, values between 2 and 5 (inclusive) suffice.
        compression_type: One of "" (no compression), "ZLIB", or "GZIP".
        max_vocab_size: The maximum number of tokens in the vocabulary.
            If not specified, there is no limit on the number of tokens.
        sequence_length: The length of each line after tokenization and
            vectorization. Pads or truncates samples to the same length.
            If not specified, there is no limit on the sequence length.
        batch_size: The size of a single batch.
        shuffle_buffer_size: The size of the buffer used to shuffle data.
        vectorization_batch_size: The number of lines to feed into the
            TextVectorization layer at a time.
    """
    def _remove_empty(x: tf.Tensor) -> tf.Tensor:
        """Filter out empty strings from the given string tensor."""
        # Convert the string tensor to a boolean tensor (where each string element
        # is replaced with True if it is not empty, or False otherwise).
        #
        # This is a nice Python trick, where empty strings (and zero and None values)
        # are treated as a False when converted to booleans.
        return tf.cast(tf.strings.length(x), bool)

    # Load the files using a TextLineDataset object, which will automatically
    # decompress the files (if needed) and split them into lines.
    text_dataset = tf.data.TextLineDataset(
        # TensorFlow can't handle pathlib.Path instances, so we convert to str.
        filenames=[str(x) for x in filenames],
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
    start_time = time.time()
    vectorizer.adapt(text_dataset.batch(vectorization_batch_size))
    vectorizer_elapsed = time.time() - start_time

    # vocabulary is a list of strings, allowing us to lookup words by encoded index.
    # By convention, the first element of the vocabulary is an empty string
    # (which is used as a padding token). Hence, word indices start at 1.
    vocabulary = vectorizer.get_vocabulary()
    vocab_size = len(vocabulary)
    logger.info(f'Created vocabulary from {filenames} (took {vectorizer_elapsed:.2f} seconds). '
                f'Vocabulary size: {vocab_size} words.')

    def _vectorize_func(x):
        # Convert string tensors to int tensors (representing words encoded by the vectorizer)
        x = tf.expand_dims(x, -1)
        return tf.squeeze(vectorizer(x))

    # A flag that tells TensorFlow to tune dataset pipeline parameters automatically.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # Vectorize text_dataset into dataset_vectors
    dataset_vectors = text_dataset.batch(1024).prefetch(AUTOTUNE).map(_vectorize_func).unbatch()

    logger.info('Creating training data...This may take a while!')
    # Make the training data from sequences
    sequences = dataset_vectors.as_numpy_iterator()
    targets, contexts, labels = make_training_data(
        sequences,
        window_size,
        n_negative_samples,
        vocab_size
    )

    logger.info('Converting training data to tf.data.Dataset...')
    # Load the training data into a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    print(dataset)
    return dataset

if __name__ == '__main__':
    import python_ta
    # python_ta.check_all(config={
    #     'extra-imports': [
    #         'typing',
    #         'pathlib',
    #         'tensorflow',
    #         'tensorflow.keras.layers.experimental.preprocessing'
    #     ],
    #     'allowed-io': [],
    #     'max-line-length': 100,
    #     'disable': ['R1705', 'C0200']
    # })

    # import python_ta.contracts
    # python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()
