"""Implementation of the Word2Vec model architecture with subsampling and negative sampling.

Based on the original Word2Vec paper ("Efficient Estimation of Word Representations in Vector Space")
and the accomponying C source code by Mikolov et. al. (https://github.com/tmikolov/word2vec).
"""

import json
import string
import itertools
from pathlib import Path
from collections import Counter
from typing import (
    Tuple,
    List,
    Dict,
    Optional,
    Union,
    Iterator,
    Generator
)

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def read_lines(filenames: List[Union[Path, str]]) -> Iterator:
    """Return an iterator of the lines in all the given files.

    Args:
        filenames: A list of strings or pathlib.Path objects containing
            the names of text files.
    """
    lines = []
    for file in filenames:
        with tf.io.gfile.GFile(file) as fp:
            lines.append(fp)
    lines = itertools.chain(*lines)
    return lines


class Tokenizer:
    """Text tokenizer.

    Instance Attributes:
        - unknown_index: Token to represent words not in the dataset.
        - max_tokens: The maximum number of tokens in the vocabulary.
            If None, there is no max on the tokens.
        - sample_threshold: A small value to offset the probabilities for
            sampling any given word. This is the "t" variable in the distribution
            given by Mikolov et. al. in their Word2Vec paper.
        - min_word_frequency: The minimum frequency for words to be included in
            the vocabulary.
        - vocabulary: A dictionary mapping a string (word) to its encoded index.
        - words: A list of strings, where the i-th element of the list
            corresponds to the word with encoded index i.
        - frequencies: A list of integers, where the i-th element of the list
            corresponds to frequency of the word with encoded index i.
        - sampling_table: A list of floats giving the probability of sampling words.
            The i-th element of the table gives the probability of sampling the word.
        - corpus_size: The number of words in the corpus.
        - vocab_size: The size of the vocabulary.

    """
    # Private Instance Attributes:
    #   - _vocabulary: A dictionary mapping a string (word) to its encoded index.
    #   - _words: A list of strings, where the i-th element of
    #       the list corresponds to the word with encoded index i.
    #   - _frequencies: a list of integers, where the i-th element of the list
    #       corresponds to frequency of the word with encoded index i.
    #   - _counter: Counts the occurences of words.
    #   - _sampling_table: A list of floats giving the probability of sampling words.
    #       The i-th element of the table gives the probability of sampling the word
    #       whose encoded index is i.
    #   - _corpus_size: The number of words in the corpus.
    #   - _remove_punctuation_trans: A translation for removing puntuation from a string.
    _vocabulary: Dict[str, int]
    _words: List[str]
    _frequencies: List[int]
    _counter: Counter
    _sampling_table: List[float]
    _corpus_size: int
    _remove_punctuation_trans: object

    def __init__(self, unknown_index: Optional[int] = -1,
                 max_tokens: Optional[int] = None,
                 min_word_frequency: Optional[int] = 0,
                 sample_threshold: Optional[float] = 1e-3) -> None:
        """Initialise this tokenizer.

        Args:
            unknown_index: Index to represent words not in the dataset.
            max_tokens: The maximum number of tokens in the vocabulary.
                If None, there is no max on the tokens. This is including
                the number of default tokens in the tokenizer.
            min_word_frequency: The minimum frequency for words to be included
                in the vocabulary.
            sample_threshold: A small value to offset the probabilities for
                sampling any given word. This is the "t" variable in the
                distribution given by Mikolov et. al. in their Word2Vec paper.
        """
        self.unknown_index = unknown_index
        self.max_tokens = max_tokens
        self.min_word_frequency = max(0, min_word_frequency)
        self.sample_threshold = sample_threshold

        self._counter = Counter()
        # Remove all punctuation except for underscores and hashtags
        self._remove_punctuation_trans = str.maketrans('', '', '!"$%&\'()*+,-./:;<=>?@[\\]^`{|}~')
        self._initialise_defaults()

    def _initialise_defaults(self, reset_counter: Optional[bool] = False) -> None:
        """Initialise this tokenizer with a default vocabulary.

        Args:
            reset_counter: Whether to reset the counter.
        """
        self._vocabulary = {}
        self._words = []
        self._frequencies = []
        self._sampling_table = []
        self._corpus_size = 0

        if reset_counter:
            self._counter = Counter()

    def reset(self) -> None:
        """Reset this tokenizer."""
        self._initialise_defaults(reset_counter=True)

    @property
    def vocabulary(self) -> Dict[str, int]:
        """Return a dictionary mapping a string (word) to its encoded index."""
        return self._vocabulary

    @property
    def words(self) -> List[str]:
        """Return a list of strings, where the i-th element of the list
        corresponds to the word with encoded index i.
        """
        return self._words

    @property
    def frequencies(self) -> List[int]:
        """Return a list of integers, where the i-th element of the list
        corresponds to frequency of the word with encoded index i.
        """
        return self._frequencies

    @property
    def sampling_table(self) -> List[float]:
        """Return a list of floats giving the probability of sampling words.
        The i-th element of the table gives the probability of sampling the word.
        """
        return self._sampling_table

    @property
    def corpus_size(self) -> int:
        """Return the number of words in the corpus."""
        return self._corpus_size

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocabulary)

    def tokenize_string(self, string: str) -> List[str]:
        """Return a generator expression of tokens.

        This removes punctuation, converts the string to lowercase,
        strips leading and trailing whitespace, and splits by spaces.
        """
        string = string.translate(self._remove_punctuation_trans)
        return string.lower().split()

    def _get_sample_probability(self, frequency: int) -> float:
        """
        Return the sample probability for a word with the given frequency.

        The sampling probabilities are generated according to the formula given by
        Mikolov et. al. in their Word2Vec paper, and closely follows the author's
        original implementation from the source code accomponying the paper.

        See: https://github.com/tmikolov/word2vec/blob/master/word2vec.c#L407

        Args:
            frequency: The frequency of the word.
        """
        # The proportion of this word in the corpus
        f = frequency / self._corpus_size
        p = (np.sqrt(f / self.sample_threshold) + 1) * (self.sample_threshold / f)
        return np.minimum(p, 1.0)

    def build(self, data: Optional[Union[str, List[str]]] = None,
              filenames: Optional[Union[str, Path, List[Union[Path, str]]]] = None,
              reset_state: Optional[bool] = False) -> None:
        """Build the vocabulary of this tokenizer from a corpus.

        Args:
            data: A string or list of strings containing text data.
            filenames: A string or pathlib.Path object, or a list of them
                containing the names of text files.
            reset_state: Whether to reset the state of the tokenizer.
        """
        # Convert the data into a list so that it is consistent.
        if data is None:
            data = []
        elif isinstance(data, str):
            data = [data]

        # Convert the filenames into a list so that it is consistent.
        if filenames is None:
            filenames = []
        elif isinstance(filenames, (str, Path)):
            filenames = [filenames]

        # Convert filenames to string since TensorFlow doesn't like path.Pathlib objects.
        filenames = [str(file) for file in filenames]

        # Combine the given data and the lines of the given files
        # into a single iterator.
        data = itertools.chain(data, read_lines(filenames))

        # Reset the state, if needed.
        if reset_state:
            self.reset()

        # Re-initialise the vocabulary to the default values.
        # This will reset everything BUT the counter, which we
        # want to keep since we want word frequency to be persistent.
        self._initialise_defaults()
        for x in data:
            self._counter.update(self.tokenize_string(x))

        tokens = self._counter.most_common(n=self.max_tokens)
        # Filter out tokens that don't appear at least min_word_frequency times in the corpus.
        tokens = [
            (word, frequency) for word, frequency in tokens
            if frequency >= self.min_word_frequency
        ]

        self._corpus_size = sum(frequency for _, frequency in tokens)
        for token, frequency in tokens:
            self._vocabulary[token] = len(self._vocabulary)
            self._words.append(token)
            self._frequencies.append(frequency)

            sample_probability = self._get_sample_probability(frequency)
            self._sampling_table.append(sample_probability)

    def encode_string(self, x: str) -> List[str]:
        """Encode a single string into an integer vector.

        Args:
            x: The string to encode.
        """
        tokens = self.tokenize_string(x)
        return [self.get_index(token) for token in tokens]

    def encode(self, inputs: List[str]) -> List[List[str]]:
        """Encode inputs.

        Args:
            inputs: A list of strings to encode.
        """
        return [self.encode_string(x) for x in inputs]

    def get_index(self, token: str) -> int:
        """Return the index of the given token.

        If the given token is out-of-the-vocabulary, the index of the
        unknown token is returned.
        """
        if token not in self._vocabulary:
            return self.unknown_index
        return self._vocabulary[token]

    def get_token(self, index: int) -> str:
        """Return the token corresponding to the given token.

        If the given index is out-of-the-vocabulary, None is returned.
        """
        if index >= len(self._words):
            return None
        return self._words[index]

    def get_frequency(self, *tokens: List[str]) -> Union[int, Tuple[int]]:
        """Return the frequency of each given token."""
        frequencies = []
        for token in tokens:
            frequencies.append(self._counter[token])

        if len(frequencies) == 1:
            return frequencies[0]
        return tuple(frequencies)

    def _get_state(self) -> dict:
        """Get the state of this tokenizer."""
        return {
            'unknown_index': self.unknown_index,
            'max_tokens': self.max_tokens,
            'sample_threshold': self.sample_threshold,
            'min_word_frequency': self.min_word_frequency,
            '_vocabulary': self._vocabulary,
            '_words': self._words,
            '_counter': self._counter,
            '_sampling_table': self._sampling_table,
            '_corpus_size': self._corpus_size
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


def apply_subsampling(sequence: tf.Tensor, sampling_table: tf.Tensor,
                      unknown_index: Optional[int] = -1) -> tf.Tensor:
    """Apply subsampling on the encoded sequence using the probabilities
    in the sampling table computed by the tokenizer.

    Return a rank-1 int tensor of word indices.

    Args:
        sequence: A rank-1 int tensor of encoded words.
        sampling_table: A rank-1 float tensor containing the probabilities
            of sampling words by frequency. The i-th element of the table
            gives the probability of sampling the word whose encoded index is i.
        unknown_index: The index of words that are not in the vocabulary.
            These are filtered out prior to subsampling.
    """
    # Filter out unknown words
    sequence = tf.boolean_mask(sequence, tf.not_equal(sequence, unknown_index))
    probabilities = tf.gather(sampling_table, sequence)
    # Random sample values from 0 to 1
    samples = tf.random.uniform(tf.shape(probabilities), 0, 1)
    # Sample if and only if the random value is less than the sample probability.
    sequence = tf.boolean_mask(sequence, tf.less(samples, probabilities))
    return sequence


def make_skipgram_pairs(sequence: tf.Tensor, window_size: int,
                        randomly_offset: Optional[bool] = True) -> tf.Tensor:
    """
    Make example pairs for a skip-gram language model.

    Return a rank-2 int tensor where each row consists of a single (target, context)
    word pair. Context words are sampled from the neighbourhood around the target word
    with a radius of the given window size.

    Args:
        sequence: A rank-1 int tensor representing a sequence of encoded word indices.
        window_size: The size of the sliding window (on each side of the
            target word) used to construct sample pairs.

            For every index i in the sequence, contexts words are chosen in
            the neigbourhood defined by [i - window_size, i + window_size].
        randomly_offset: Whether to randomly offset the window size.

    Preconditions:
        - window_size > 0

    >>> sequence = tf.convert_to_tensor([4, 3, 5, 5, 2], dtype=tf.int64)
    >>> expected = [[4, 3], [4, 5],\
                    [3, 4], [3, 5], [3, 5],\
                    [5, 4], [5, 3], [5, 5], [5, 2],\
                    [5, 3], [5, 5], [5, 2],\
                    [2, 5], [2, 5]]
    >>> actual = make_skipgram_pairs(sequence, 2, randomly_offset=False)
    >>> tf.reduce_all(tf.equal(actual, expected)).numpy()
    True
    """
    n = tf.size(sequence)

    def _make_pairs(index: int, x: tf.TensorArray) -> Tuple[int, tf.TensorArray]:
        """Make (target, context) pairs for a given target word.

        Returns the next iteration index (variable), and a TensorArray
        containing the output values.

        Args:
            index: The index of the target word in the sequence tensor.
            x: Collection holding the output values.
        """
        if randomly_offset:
            shift = tf.random.uniform((), maxval=window_size, dtype=tf.int32)
        else:
            shift = 0

        # Calculate indices of context words to the left and right of the target.
        left = tf.range(tf.maximum(0, index - window_size + shift), index)
        right = tf.range(index + 1, tf.minimum(n, index + 1 + window_size - shift))
        # Concatenate left and right tensors
        contexts = tf.concat([left, right], axis=0)
        contexts = tf.gather(sequence, contexts)
        # Create (target, context) pairs
        targets = tf.fill(tf.shape(contexts), sequence[index])
        pairs = tf.stack([targets, contexts], axis=1)
        # Output values
        return index + 1, x.write(index, pairs)

    # Placeholder array that will store the output values of _make_pairs
    x = tf.TensorArray(tf.int64, size=n, infer_shape=False)
    # Outputs a list of tensors for each loop variable.
    _, x = tf.while_loop(lambda i, _: i < n,  # Stop condition
                         _make_pairs,  # Body
                         [0, x],  # Loop variables
                         back_prop=False)
    # Concat outputs for each loop variable into a single tensor
    outputs = tf.cast(x.concat(), tf.int64)
    # Ensure shape of the output
    outputs.set_shape((None, 2))
    return outputs


def make_dataset(filenames: List[Union[Path, str]], tokenizer: Tokenizer,
                 window_size: Optional[int] = 5, batch_size: Optional[int] = 32,
                 epochs: Optional[int] = 1) -> tf.data.Dataset:
    """
    Make a dataset for training the Word2Vec model using the given tokenizer.

    Return a tf.data.Dataset consisting of feature-label tensor tuples of the
    form (features, labels, iter_progress), where iter_progress is the iteration
    progress (i.e. percent complete) at each sample.

    The shape of the dataset is (
        (batch_size,),
        (batch_size,),
        (batch_size,)
    ).

    Args:
        filenames: A list of strings or pathlib.Path objects containing
            the names of text files.
        tokenizer: The tokenizer to use to encode the text data.
        window_size: The size of the sliding window used to construct sample pairs.
        batch_size: The size of a single batch.
        epochs: The number of times the dataset is iterated over during training.

    Preconditions:
        - window_size > 0
        - batch_size > 0
    """
    # Convert filenames to string since TensorFlow doesn't like path.Pathlib objects.
    filenames = [str(file) for file in filenames]

    # Get the total number of lines that we have to iterate over
    total_lines = sum(len(list(tf.io.gfile.GFile(file))) for file in filenames) * epochs

    def _load_lines_generator() -> List[int]:
        """Generator function for getting lines of the dataset."""
        lines = read_lines(filenames)
        for line in lines:
            yield tokenizer.encode_string(line)

    dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_generator(_load_lines_generator, tf.int64, output_shapes=[None]),
        # A tensor containing the iteration progress at each sample.
        tf.data.Dataset.from_tensor_slices(tf.range(total_lines) / total_lines)
    ))

    # Convert the sampling table to a tensor so it can be used by TensorFlow.
    sampling_table = tf.cast(tf.constant(tokenizer.sampling_table), tf.float32)
    # Apply subsampling on each sequence
    dataset = dataset.map(lambda sequence, iter_progress: (
        apply_subsampling(sequence, sampling_table), iter_progress)
    )
    # Filter out sequences that don't have at least 2 tokens
    # We can't slide a window on a single token!
    dataset = dataset.filter(lambda sequence, iter_progress:
        tf.greater(tf.size(sequence), 1)
    )
    # Make skip-gram pairs
    dataset = dataset.map(lambda sequence, iter_progress:
        (make_skipgram_pairs(sequence, window_size), iter_progress)
    )
    # Now, we have training (target, context) word pairs
    # instead of int tensor sequences.
    #
    # Transforms the dataset into a list of tuples (pairs, iter_progress)
    # where pairs is a tuple of rank-1 int tensors and iter_progresses is
    # a rank-1 float tensor.
    dataset = dataset.map(lambda samples, iter_progress:
        # We want an inter_progress entry for each sample pair.
        # This will be used to calculate the learning rate of the model.
        (samples, tf.fill(tf.shape(samples)[:1], iter_progress))
    )
    # Flatten the dataset into a contiguous collection of (sample, iter_progress) tuples.
    dataset = dataset.flat_map(lambda samples, iter_progress:
        # from_tensor_slices takes tensors and transforms it into a new
        # dataset whose elements are slices of the tensor.
        # So, we will get a new dataset with nested lists whose elements
        # are tuples of the form: (sample, iter_progress).
        tf.data.Dataset.from_tensor_slices((samples, iter_progress))
    )
    # Batch the dataset into tensors with shape (batch_size, 2,).
    dataset = dataset.batch(batch_size, drop_remainder=True)

    def _split_sample_pairs(x: tf.Tensor, iter_progress: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Split the sample pair tensors into
        (features, labels, iter_progress) tensor tuples.
        """
        # AutoGraph requires us to set the shape of tensors when
        # it can't automatically infer it from context.
        x.set_shape((batch_size, 2))

        features = tf.squeeze(x[:, :1], axis=1)
        labels = tf.squeeze(x[:, 1:], axis=1)
        iter_progress = tf.cast(iter_progress, tf.float32)
        return features, labels, iter_progress

    dataset = dataset.map(_split_sample_pairs)
    return dataset


class Word2Vec(tf.keras.Model):
    """Word2Vec model.

    Instance Attributes:
        - hidden_size: The number of units in the hidden layer.
            This is also the dimensionality of the embedding vectors.
        - batch_size: The size of a single batch.
        - n_negative_samples: The number of negative samples to generate per
            positive context word. Mikolov, et. al. showed that for small
            datasets, values between 5 and 20 (inclusive) work best, whereas
            for large datasets, values between 2 and 5 (inclusive) suffice.
        - lambda_power: Used to skew the probability distribution when sampling words.
    """
    # Private Instance Attributes:
    #   - _tokenizer: The tokenizer containing the vocabulary.
    #   - _bias: Whether to add a bias term.
    _tokenizer: Tokenizer

    def __init__(self, tokenizer: Tokenizer, hidden_size: Optional[int] = 256,
                 batch_size: Optional[int] = 256, n_negative_samples: Optional[int] = 5,
                 lambda_power: Optional[float] = 0.75, bias: Optional[bool] = True) -> None:
        """Initialise this Word2Vec model.

        Args:
            tokenizer: The tokenizer containing the vocabulary.
            hidden_size: The number of units in the hidden layer.
                This is also the dimensionality of the embedding vectors.
            batch_size: The size of a single batch.
            n_negative_samples: The number of negative samples to generate per
                positive context word. Mikolov, et. al. showed that for small
                datasets, values between 5 and 20 (inclusive) work best, whereas
                for large datasets, values between 2 and 5 (inclusive) suffice.
            lambda_power: Used to skew the probability distribution when sampling words.
            bias: Whether to add a bias term.
        """
        # Call the superclass initialiser
        super().__init__()

        self._tokenizer = tokenizer
        self._bias = bias

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_negative_samples = n_negative_samples
        self.lambda_power = lambda_power

        self._create_weights()

    def _create_weights(self):
        """Create weights for the model."""
        vocab_size = self._tokenizer.vocab_size
        # Weights for the first hidden layer ('projects' inputs into the embedding vector space).
        # These are used to retrieve embedding vectors.
        self.add_weight('proj', shape=(vocab_size, self.hidden_size),
                        # Initialise weights by sampling from a uniform distribution
                        # in the range (-1 / (2 * self.hidden_size), 1 / (2 * self.hidden_size)).
                        initializer=tf.keras.initializers.RandomUniform(
                            minval=-0.5 / self.hidden_size,
                            maxval=0.5 / self.hidden_size
                        ))

        # Weights for the second hidden layer (internal weights for projecting the embedding
        # space back into the vocabulary output space). These are only used for training.
        self.add_weight('proj_out', shape=(vocab_size, self.hidden_size),
                        # Initialise weights by sampling from a uniform distribution
                        # in the range (-0.1, 0.1).
                        initializer=tf.keras.initializers.RandomUniform(
                            minval=-0.1,
                            maxval=0.1
                        ))

        # Bias term to add when taking the dot product of the hidden layers.
        self.add_weight('bias', shape=(vocab_size,),
                        # Initialize bias with zeroes.
                        initializer=tf.keras.initializers.Zeros())

    def call(self, inputs: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """Run a forward pass on the model.

        Return a float tensor with shape (self.batch_size, self.n_negative_samples + 1)
        represesnting the cross entropy loss for the given features and chosen negative samples.

        Args:
            inputs: An int tensor with shape (self.batch_size,) containing
                the features to the model.
            labels: An int tensor with shape (self.batch_size,) containing
                the labels (correct output) for the given feature inputs.

        >>> import math
        >>> tf.random.set_seed(69)

        >>> tokenizer = Tokenizer()
        >>> tokenizer.build('David is really cool!')
        >>> model = Word2Vec(tokenizer, hidden_size=tokenizer.vocab_size,\
                             batch_size=1, n_negative_samples=1)

        >>> inputs = tf.convert_to_tensor([2], dtype=tf.int64)
        >>> labels = tf.convert_to_tensor([1], dtype=tf.int64)

        >>> actual = tf.reduce_sum(tf.squeeze(model(inputs, labels))).numpy()
        >>> expected = 1.3795923
        >>> math.isclose(actual, expected)
        True
        """
        # Create n_negative_samples for each positve (target, context) word-pair,
        # by sampling random words from the vocabulary (excluding the context word).
        #
        # A negative sample refers to a (target, not_context) word-pair where not_context
        # is NOT the context word of the positive sample (i.e. not_context != context).
        #
        # The intuition towards negative sampling, as proposed by Mikolov et. al.
        # is to sample words from the vocabulary from a distribution designed to favour
        # more frequent words. The probability of sampling a word i is given by:
        #   P(w_i) = [f(w_i)^lambda] / sum(f(w_j)^lambda for j = 0 to n),
        # where n is the vocabulary size, f : N -> N, gives the frequency of each word
        # in the vocabulary, and lambda is a hyperparameter (set to 3/4 in the paper).
        #
        # Sample a random word (index) from the range [0, vocab_size) excluding the
        # elements of the true_classes tensor (in our case, the context word).
        #
        # Returns an int tensor with shape (n_negative_samples,) containing the sampled values.
        negative_samples = tf.random.fixed_unigram_candidate_sampler(
            true_classes=tf.expand_dims(labels, 1),
            num_true=1,
            num_sampled=self.batch_size * self.n_negative_samples,
            unique=True,  # Sample without replacement
            range_max=self._tokenizer.vocab_size,
            distortion=self.lambda_power,
            unigrams=self._tokenizer.frequencies
        )
        # Get candidates
        negative_samples = negative_samples.sampled_candidates
        # Reshape negative samples into a matrix
        negative_samples = tf.reshape(negative_samples, (self.batch_size, self.n_negative_samples))

        # Get weights
        proj, proj_out, bias = self.weights
        # Project inputs into embedding space
        proj_inputs = tf.gather(proj, inputs)
        # Project lables into output space
        proj_labels = tf.gather(proj_out, labels)

        # Compute logits
        label_logits = tf.reduce_sum(tf.multiply(proj_inputs, proj_labels), 1)

        negative_samples_proj = tf.gather(proj_out, negative_samples)
        # Transpose the negative sample matrix so that the weights and inputs are flipped.
        ns_proj_transpose = tf.transpose(negative_samples_proj, (0, 2, 1))
        # Einstein summation (einsum) is a compact way of expressing element-wise computation.
        # In this case, we take the sum of A_ijk * B_ikl for all i,j,k,l to compute a new tensor C_il.
        # This can be thought of as nested summations:
        #   C = sum(
        #           sum(
        #               sum(
        #                   sum(
        #                       sum(A_ijk * B_ikl = C_il for all l)
        #               for all k)
        #           for all j)
        #       for all i).
        # Or concisely written as ijk,ikl->il.
        pred_logits = tf.einsum('ijk,ikl->il', tf.expand_dims(proj_inputs, 1), ns_proj_transpose)
        # Add bias if needed
        if self._bias:
            label_logits += tf.gather(bias, labels)
            pred_logits += tf.gather(bias, negative_samples)

        # Compute cross entropy
        label_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(label_logits), logits=label_logits
        )
        pred_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(pred_logits), logits=pred_logits
        )
        # Compute loss
        return tf.concat([
            # label_cross_entropy is a tensor with shape (batch_size,) whereas
            # pred_cross_entropy is a tensor with shape (batch_size, n_negative_samples).
            # We need to add another dimension to the former so the shapes match up.
            tf.expand_dims(label_cross_entropy, 1),
            pred_cross_entropy
        ], axis=1)

    def train(self, dataset: tf.data.Dataset, logdir: Union[str, Path],
              initial_lr: Optional[float] = 0.025, target_lr: Optional[float] = 1e-4,
              log_frequency: Optional[int] = 1000, save_frequency: Optional[int] = 10000,
              max_checkpoints: Optional[int] = 4, show_progress_bar: Optional[bool] = True)\
            -> None:
        """Train the model.

        Args:
            dataset: The dataset to train on.
            logdir: The output directory.
            initial_lr: The initial learning rate.
            target_lr: The target learning rate.
            log_frequency: The frequency at which to log stats, in global steps.
            save_frequency: The frequency at which to save the model.
            max_checkpoints: The maximum number of checkpoints to keep at a time.
                If not positive, then there is no maximum on the number of saved checkpoints.
            show_progress_bar: Whether to show a progress bar while the jobs run.
        """
        # Create output directory
        logdir.mkdir(parents=True, exist_ok=True)

        # Create summary writers (for logging to tensorboard)
        summary_writer = tf.summary.create_file_writer(str(logdir))

        # Use stochastic gradient descent without learning rate
        # We will apply the learning rate ourselves, with a custom decay.
        optimizer = tf.keras.optimizers.SGD(1.0)
        input_signature = [
            tf.TensorSpec(shape=(self.batch_size,), dtype=tf.int64),
            tf.TensorSpec(shape=(self.batch_size,), dtype=tf.int64),
            tf.TensorSpec(shape=(self.batch_size,), dtype=tf.float32)
        ]

        @tf.function(input_signature=input_signature)
        def _train_step(inputs: tf.Tensor, labels: tf.Tensor, iter_progress: tf.Tensor) \
                -> Tuple[tf.Tensor, tf.Tensor]:
            """Train the word2vec model for a single step.

            This function computes the training loss and performs backprop using an optimizer.

            Args:
                inputs: Input features to the model (targets).
                labels: Correct labels for the model (contexts).
            """
            loss = self.call(inputs, labels)
            # Compute gradients
            gradients = tf.gradients(loss, self.trainable_variables)

            # Linearly interpolate between the initial and target learning rate.
            t = iter_progress[0]
            learning_rate = initial_lr * (1 - t) + target_lr * t
            learning_rate = tf.maximum(learning_rate, target_lr)

            # Apply learning rate to the gradients for each trainable variable/weight.
            # In our case, this is for the proj, proj_out, and bias layer.
            for i in range(len(self.trainable_variables)):
                if hasattr(gradients[i], '_values'):
                    gradients[i]._values *= learning_rate
                else:
                    gradients[i] *= learning_rate

            # Apply gradients for backprop
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return loss, learning_rate

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
        manager = tf.train.CheckpointManager(checkpoint, directory=logdir, max_to_keep=max_checkpoints)

        average_loss = 0
        with tqdm(dataset, disable=not show_progress_bar) as progress_bar:
            for global_step, (inputs, labels, iter_progress) in enumerate(progress_bar):
                loss, learning_rate = _train_step(inputs, labels, iter_progress,)
                average_loss += loss.numpy().mean()
                if global_step % log_frequency == 0:
                    if global_step > 0:
                        average_loss /= log_frequency

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

                if global_step > 0 and global_step % save_frequency == 0:
                    manager.save()


if __name__ == '__main__':
    import python_ta
    # python_ta.check_all(config={
    #     'extra-imports': [
    #         'json',
    #         'string',
    #         'itertools',
    #         'pathlib',
    #         'collections',
    #         'typing',
    #         'numpy',
    #         'tensorflow',
    #         'tqdm'
    #     ],
    #     'max-line-length': 100,
    #     'disable': ['R1705', 'C0200', 'E9998']
    # })

    # import doctest
    # doctest.testmod()
