"""Implementation of the Word2Vec model architecture with subsampling and negative sampling."""

import time
import json
import string
import random
from pathlib import Path
from collections import Counter
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
                       show_progress_bar: Optional[bool] = True,
                       progress_bar_total: Optional[int] = None) \
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
        progress_bar_total: Total number of iterations for the progress bar.
            This is purely aesthetic.

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

    for sequence in tqdm(sequences, disable=not show_progress_bar, total=progress_bar_total):
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


def read_lines(filenames: List[Union[Path, str]]) -> List[str]:
    """Return a list of non-empty lines from the given files."""
    all_lines = []
    for file in filenames:
        with open(file) as fp:
            lines = fp.read().splitlines()
            # Filter out empty lines
            #
            # This is a nice Python trick, where empty strings
            # are treated as a False when converted to booleans.
            all_lines.extend([line for line in lines if line])
    return all_lines


class Tokenizer:
    """Text tokenizer.

    By convention, the first element of the vocabulary is an empty string
    and is used a padding token. Word indices are one-based. And, the vocabulary
    is sorted by word frequency, where the most-common word has index 2, the
    second-most common word has index 3, and so on.

    Instance Attributes:
        - pad_token: Token to represent non-words. A padding token.
        - unknown_token: Token to represent words not in the dataset.
        - max_tokens: The maximum number of tokens in the vocabulary.
            If None, there is no max on the tokens.
    """
    # Private Instance Attributes:
    #   - _vocabulary: A dictionary mapping a string (word) to its encoded index.
    #   - _inverse_vocabulary: A list of strings, where the i-th element of
    #           the list corresponds to the word with encoded index i.
    #   - _counter: Counts the occurences of words.
    #   - _remove_punctuation_trans: A translation for removing puntuation from a string.
    _vocabulary: Dict[str, int]
    _inverse_vocabulary: List[str]
    _counter: Counter
    _remove_punctuation_trans: object

    def __init__(self, pad_token='', unknown_token='UNK',
                 max_tokens: Optional[int] = None) -> None:
        """Initialize this tokenizer.

        Args:
            pad_token: Token to represent non-words. A padding token.
                This token has index 0 in the vocabulary.
            unknown_token: Token to represent words not in the dataset.
                This token has index 1 in the vocabulary.
            max_tokens: The maximum number of tokens in the vocabulary.
                If None, there is no max on the tokens. This is including
                the number of default tokens in the tokenizer.
        """
        self._counter = Counter()
        self._remove_punctuation_trans = str.maketrans('', '', string.punctuation)

        self.pad_token = pad_token
        self.unknown_token = unknown_token
        self.max_tokens = max_tokens

        self._initialise_defaults()

    @property
    def vocabulary(self) -> Dict[str, int]:
        """Return a dictionary mapping a string (word) to its encoded index."""
        return self._vocabulary

    @property
    def inverse_vocabulary(self) -> List[str]:
        """Return a list of strings, where the i-th element of the
        list corresponds to the word with encoded index i.
        """
        return self._inverse_vocabulary

    def _initialise_defaults(self) -> None:
        """Initialise the default vocabulary and inverse vocabulary."""
        self._vocabulary = {self.pad_token: 0, self.unknown_token: 1}
        self._inverse_vocabulary = [self.pad_token, self.unknown_token]

    def reset(self) -> None:
        """Reset this tokenizer."""
        self._initialise_defaults()
        self._counter = Counter()

    def update(self, data: Union[str, List[str]],
               reset_state: Optional[bool] = False) -> None:
        """Update this tokenizer with new text data.

        Args:
            data: A string, list of strings, or string tensor
                containing text data to update the tokenizer with.
            reset_state: Whether to reset the state of the tokenizer.
        """
        # Reset the sate if needed
        if reset_state:
            self.reset()

        if isinstance(data, str):
            # Convert the data into a list so that it is consistent.
            data = [data]

        tokens = []
        for string in data:
            tokens.extend(self._tokenize_string(string))

        for token in tokens:
            # Update the counter
            self._counter[token] += 1

        self._initialise_defaults()

        n = self.max_tokens
        if n is not None:
            n -= len(self._vocabulary)

        most_common_tokens = self._counter.most_common(n)
        for token, _ in most_common_tokens:
            self._vocabulary[token] = len(self._vocabulary)
            self._inverse_vocabulary.append(token)

    def _tokenize_string(self, string: str) -> List[str]:
        """Return a list of tokens.

        This removes punctuation, converts the string
        to lowercase, and splits by spaces.
        """
        string = string.translate(self._remove_punctuation_trans)
        return string.lower().split()

    def _add_tokens(self, *tokens: List[str]) -> None:
        """Add the given tokens to the tokenizer, in the order of the list.

        If the token already exists in the vocabulary, then it is skipped,
        and its index is NOT changed.
        """
        for token in tokens:
            # Update the counter
            self._counter[token] += 1

            if token in self._vocabulary:
                continue

            self._vocabulary[token] = len(self._vocabulary)
            self._inverse_vocabulary.append(token)

    def get_frequency(self, *tokens: List[str]) -> Union[int, Tuple[int]]:
        """Return the frequency of each given token.
        """
        frequencies = []
        for token in tokens:
            frequencies.append(self._counter[token])

        if len(frequencies) == 1:
            return frequencies[0]
        return tuple(frequencies)

    def get_index(self, token: str) -> int:
        """Return the index of the given token.
        If the given token is out-of-the-vocabulary, the index of the
        unknown token is returned.
        """
        if token not in self._vocabulary:
            return self._vocabulary[self.unknown_token]
        return self._vocabulary[token]

    def get_token(self, index: int) -> str:
        """Return the token corresponding to the given token.
        If the given index is out-of-the-vocabulary, the unknown token is returned.
        """
        if index >= len(self._inverse_vocabulary):
            return self.get_index(self.unknown_token)
        return self._inverse_vocabulary[index]

    def encode(self, inputs: Union[str, tf.Tensor, List[str], Tuple[str], np.ndarray],
               output_length: Optional[int] = None) -> tf.Tensor:
        """Encode inputs.

        Args:
            inputs: The strings to encode.
            output_length: The length (dimensionality) of the encoded vectors.
                Pads or truncates vectors to the same length.
                If unspecified, vectors are padded to the max length.
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        if isinstance(inputs, (list, tuple, np.ndarray)):
            inputs = tf.convert_to_tensor(inputs)

        outputs = []
        flattened = tf.unstack(tf.reshape(inputs, [-1]))
        for x in flattened:
            value = x.numpy().decode()
            outputs.append(self._encode_string(value))

        # Compute length
        if output_length is None:
            output_length = max(len(x) for x in outputs)

        # Pad or truncate encoded vectors
        pad_token_index = self.get_index(self.pad_token)
        for i, x in enumerate(outputs):
            n = len(x)
            if output_length < n:
                # Truncate the output to output_length
                outputs[i] = x[:output_length]
            else:
                # Pad the output
                pad_length = output_length - n
                outputs[i] = x + [pad_token_index] * pad_length

        outputs = tf.convert_to_tensor(outputs, dtype=tf.int64)
        if outputs.shape[0] == 1:
            outputs = tf.squeeze(outputs, 0)
        return tf.stack(outputs)

    def _encode_string(self, input: str) -> List[str]:
        """Encode a string.

        Args:
            input: The string to encode.
        """
        tokens = self._tokenize_string(input)
        return [self.get_index(token) for token in tokens]

    def _get_state(self) -> dict:
        """Get the state of this tokenizer."""
        return {
            '_vocabulary': self._vocabulary,
            '_inverse_vocabulary': self._inverse_vocabulary,
            '_counter': self._counter,
            'pad_token': self.pad_token,
            'unknown_token': self.unknown_token,
            'max_token': self.max_tokens
        }

    def _load_state(self, state: dict) -> None:
        """Initialise this tokenizer from a state.

        It is assumed that the state dictionary is of the
        form returned by the _get_state method.
        """
        self.reset()
        for key, value in state.items():
            if key == '_counter':
                value = Counter(value)
            setattr(self, key, value)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save the state of this tokenizer to a file."""
        with open(filepath, 'w+') as file:
            json.dump(self._get_state(), file)

    def load(self, filepath: Union[str, Path]) -> None:
        """Load a tokenizer from file."""
        with open(filepath) as file:
            self._load_state(json.load(file))


def load_dataset(filenames: List[Union[Path, str]],
                 window_size: int, n_negative_samples: int,
                 max_vocab_size: Optional[int] = None,
                 sequence_length: Optional[int] = None,
                 batch_size: Optional[int] = 1024,
                 shuffle_buffer_size: Optional[int] = 10000) \
        -> tf.data.Dataset:
    """
    Load a corpus from the given files as a tf.data.Dataset object.

    Args:
        filenames: A list of filenames to load from.
            It is expected that these are text files.
        window_size: The size of the sliding window used to construct sample pairs.
        n_negative_samples: The number of negative samples to generate per
            positive context word. Mikolov, et. al. showed that for small
            datasets, values between 5 and 20 (inclusive) work best, whereas
            for large datasets, values between 2 and 5 (inclusive) suffice.
        max_vocab_size: The maximum number of tokens in the vocabulary.
            If not specified, there is no limit on the number of tokens.
        sequence_length: The length of each line after tokenization and
            vectorization. Pads or truncates samples to the same length.
            If not specified, there is no limit on the sequence length.
        batch_size: The size of a single batch.
        shuffle_buffer_size: The size of the buffer used to shuffle data.
    """
    # TODO: Gotta optimize this function!! Specifically with generating sequences.
    # Loading in a 250MB dataset takes 5 hours! Might want to look into parallelising
    # the sequence generation, or just reducing the dataset size.

    # Load the files and read non-empty lines.
    lines = read_lines(filenames)

    # The Tokenizer will create a vocabulary from strings.
    # In a nut shell, it standardises the data (lower and strip punctuation),
    # tokenizes it, and then assigns an integer index to each word.
    tokenizer = Tokenizer(max_tokens=max_vocab_size)

    # Apply the data to tokenizer to build a vocabulary.
    start_time = time.time()
    tokenizer.update(lines)
    tokenizer_elapsed = time.time() - start_time

    vocab_size = len(tokenizer.inverse_vocabulary)
    logger.info(
        f'Created vocabulary from {filenames} (took {tokenizer_elapsed:.2f} seconds). '
        f'Vocabulary size: {vocab_size} words.'
    )

    def _encode_func(x: tf.Tensor) -> tf.Tensor:
        """Convert string tensors to int tensors using the tokenizer."""
        x = tf.expand_dims(x, -1)
        # Wrap the tokenizer encode function in a TensorFlow eager op.
        # This lets us evaluate tensors eagerly, which the tokenizer needs to do.
        return tf.squeeze(tf.py_function(
            tokenizer.encode, [x], tf.int64
        ))

    # A flag that tells TensorFlow to tune dataset pipeline parameters automatically.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset_lines = tf.data.Dataset.from_tensor_slices(lines)
    # Encode lines into sequences
    dataset_sequences = dataset_lines.batch(1024).prefetch(AUTOTUNE).map(_encode_func).unbatch()
    sequences = dataset_sequences.as_numpy_iterator()

    # Make the training data from sequences
    logger.info('Creating training data...This may take a while!')
    targets, contexts, labels = make_training_data(
        sequences,
        window_size,
        n_negative_samples,
        vocab_size,
        progress_bar_total=len(lines)
    )

    logger.info('Converting training data to tf.data.Dataset...')
    # Load the training data into a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
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
