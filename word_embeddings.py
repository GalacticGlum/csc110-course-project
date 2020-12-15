"""Functionality for loading, manipulating, and searching word embedding vector spaces."""
from pathlib import Path
from typing import (
    Optional,
    Tuple,
    List,
    Dict
)

import numpy as np
from sklearn import decomposition, neighbors
from suffix_trees.STree import STree as SuffixTree
from logger import logger


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Return the cosine similarity of the two given vectors.

    Preconditions:
        - u.shape == v.shape and u.ndim == 1
    """
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


class WordEmbeddings:
    """Contains the learned word embeddings.
    Represents a discretized vector space embedding the words of a vocabulary.

    Instance Attributes:
        - weights: A matrix with shape (vocab_size, n) where n is the dimensionality
            of the embedding vectors (i.e. the number of components). The i-th row of
            the matrix should corresponding to the embedding vector for the word with
            encoded index i.
        - words: A list of strings, where the i-th element of the list
                corresponds to the word with encoded index i.
        - name_metadata: The name of the word embeddings checkpoint.
    """
    weights: np.ndarray
    words: List[str]
    weights_filepath: Path
    vocab_filepath: Path
    name_metadata: str

    # Private Instance Attributes:
    #   - _vocabulary: A dictionary mapping a word to its index.
    #   - _pca: Fitted sklearn PCA object.
    #   - _reduced_weights: Word embeddings reduced to a lower dimensional space.
    #   - _all_words: A space-separated string containing all the words in the vocabulary.
    #   - _suffix_tree: A suffix tree for fast searching of the vocabulary.
    #   - _nearest_neighbours: A nearest neighbours model for finding most similar embeddings.
    _vocabulary: Dict[str, int]
    _pca: Optional[decomposition.PCA] = None
    _reduced_weights: Optional[np.ndarray] = None
    _all_words: str
    _suffix_tree: SuffixTree
    _nearest_neighbours: neighbors.NearestNeighbors

    def __init__(self, weights_filepath: Optional[Path] = None,
                 vocab_filepath: Optional[Path] = None,
                 checkpoint_filepath: Optional[Path] = None,
                 weights: Optional[np.ndarray] = None,
                 words: Optional[List[str]] = None,
                 name_metadata: Optional[str] = None,
                 suffix_tree: Optional[bool] = True,
                 nearest_neighbours: Optional[bool] = True) -> None:
        """Initialize this word embeddings.

        Requires one of:
            - weights and words
            - checkpoint_filepath
            - weights_filepath and vocab_filepath

        Args:
            weights_filepath: Filepath to a numpy file containing trained model weights
                for the projection layer, corresponding to the learned embedding vectors
                of the words.
            vocab_filepath: A text file containing words of the vocabulary sorted in increasing
                order of the index, separated by new lines (i.e. the word on line 1 indicates
                the word with encoded index 0, and so on).
            checkpoint_filepath: Path to a checkpoint directory containing a numpy file with
                the trained embedding weights (proj_weights.npy) and a text file with the model
                vocabulary (vocab.txt).
            weights: A matrix with shape (vocab_size, n) where n is the dimensionality
                of the embedding vectors (i.e. the number of components). The i-th row of
                the matrix should corresponding to the embedding vector for the word with
                encoded index i.
            words: A list of strings, where the i-th element of the list
                corresponds to the word with encoded index i.
            name_metadata: The name of the word embeddings checkpoint.
            suffix_tree: Whether to build the suffix tree.
            nearest_neighbours: Whether to build the nearest neighbours model.
        """
        if weights and words:
            self.words = words
            self.weights = weights
        elif weights_filepath and vocab_filepath:
            # Load weights and words from the given paths
            self.weights, self.words = WordEmbeddings._load_weights_and_vocab(
                weights_filepath, vocab_filepath)
        elif checkpoint_filepath:
            # Load weights and words from the checkpoint.
            # We assume the weights file is called proj_weights
            # and the words file is called vocab.txt.
            self.weights, self.words = WordEmbeddings._load_weights_and_vocab(
                checkpoint_filepath / 'proj_weights.npy',
                checkpoint_filepath / 'vocab.txt'
            )
        else:
            raise ValueError('Requires one of (weights, words) / checkpoint_filepath / '
                             '(weights_filepath, vocab_filepath). Got None for all.')

        self.name_metadata = name_metadata
        self._vocabulary = {word: i for i, word in enumerate(self.words)}
        self._pca = None
        self._reduced_weights = None

        if suffix_tree:
            self._build_suffix_tree()
        if nearest_neighbours:
            self._build_nearest_neighbours()

    @staticmethod
    def _load_weights_and_vocab(weights_filepath: Path, vocab_filepath: Path) \
            -> Tuple[np.ndarray, List[str]]:
        """Loads the weights and vocabulary into a numpy array and list respectively.
        Return a 2-tuple containing the weights and words.
        """
        with open(vocab_filepath) as file:
            words = file.read().splitlines()
        weights = np.load(weights_filepath)
        return (weights, words)

    def _build_suffix_tree(self) -> None:
        """Build a suffix tree from the vocabulary."""
        logger.info('Building suffix tree for embedding (%s)' % str(self))
        self._all_words = ' '.join(self.words)
        self._suffix_tree = SuffixTree(self._all_words)
        logger.info('Finished building suffix tree!')

    def _build_nearest_neighbours(self) -> None:
        """Build a nearest neighbour searcher from the embedding vectors."""
        logger.info('Building nearest neighbours for embeddings (%s)' % str(self))

        # We use a KNN model to perform embedding similarity search quickly.
        # The goal is to find the most similar embedding vectors based on their cosine similarity.
        # However, while KNN does not support the cosine metric, by normalizing the embedding
        # vectors, we can use a KNN on Euclidean distance to find the most similar vectors, and
        # we will get the same ordering as we would if we used cosine similarity.
        self._nearest_neighbours = neighbors.NearestNeighbors(n_neighbors=10)
        # Normalized the weights to have unit norm
        normalized_weights = self.weights / np.linalg.norm(self.weights, axis=-1, keepdims=True)
        self._nearest_neighbours.fit(normalized_weights)
        logger.info('Finished building nearest neighbours!')

    def most_similar(self, word: Optional[str] = None, k: Optional[int] = 10) \
            -> List[Tuple[str, float]]:
        """Finds the most similar words to the given word, based on the cosine similarity.

        Return a list of 2-element tuple of the word and similarity,
        sorted in decreasing order of the similarity.

        If the given word is not in the vocabulary, an empty list is returned.

        Args:
            word: The search word. Required if vector is not specified.
            k: The number of most similar words to return.
                If unspecified, all words in the vocabulary are returned.
            vector: A vector with the same number of dimensions as the vector
                embeddings to search instead of a word.
        """
        if word not in self._vocabulary:
            return []

        # Default to the vocab size
        # Clamp the given value of k to be in the range [0, vocab_size].
        vocab_size = len(self.words)
        # We get the k + 1 nearest neighbours since the model gives back the input as well.
        k = max(min((k or vocab_size) + 1, vocab_size), 0)

        # Lookup the embedding vector
        word_index = self._vocabulary[word]
        vector = self.weights[word_index]
        # Get the nearest neighbours
        # The KNN returns a numpy array with shape (batch_size, vector_size),
        # but in our case the batch size is just 1 (the single embedding vector input).
        _, indices = self._nearest_neighbours.kneighbors([vector], n_neighbors=k)

        most_similar = [(
            self.words[index],
            # Recompute the distance, but using cosine similarity.
            cosine_similarity(vector, self.weights[index])
        ) for index in indices[0] if index != word_index]

        return most_similar

    def get_vector(self, word: str) -> np.ndarray:
        """Return the embedding vector for the given word."""
        return self.weights[self._vocabulary[word]]

    def pca(self, spherize: Optional[bool] = True, top_k_components: Optional[int] = 10,
            force_rebuild: Optional[bool] = False) -> Tuple[decomposition.PCA, np.ndarray]:
        """Get/build the PCA for this word embedding vector space.
        Return the sklearn.decomposition.PCA instance, and the lower-dimension weights.

        Args:
            spherize: Whether to spherize the data. This shifts the data by the centroid,
                and normalizes embeddings to have unit norms.
            top_k_components: Number of components to reduce the vector space to.
            force_rebuild: Whether to rebuild the PCA if it has already been computed.
        """
        # If we force_rebuild is False and we have already computed the PCA,
        # then we don't need to do anything.
        if self._pca is not None and self._reduced_weights is not None and not force_rebuild:
            return self._pca, self._reduced_weights

        if spherize:
            # Shift each point by the centroid
            centroid = np.mean(self.weights, axis=0)
            weights = self.weights - centroid
            # Normalize data to unit norms
            weights = weights / np.linalg.norm(weights, axis=-1, keepdims=True)
        else:
            weights = self.weights

        self._pca = decomposition.PCA(n_components=top_k_components)
        self._reduced_weights = self._pca.fit_transform(weights)
        return self._pca, self._reduced_weights

    def search_words(self, query: str) -> List[str]:
        """Return a list of strings that contain the query string."""
        matches = self._suffix_tree.find_all(query.lower())
        words = []
        for index in matches:
            # Find the last and next space in the string, to get the whole word.
            i = self._all_words.rfind(' ', 0, index)
            j = self._all_words.find(' ', index)
            words.append(self._all_words[i + 1:j])

        return words

    def __getitem__(self, word: str) -> np.ndarray:
        """Return the embedding vector for the given word."""
        return self.get_vector(word)

    def __str__(self) -> str:
        """Return a string representation of this word embedding vector space."""
        if self.name_metadata is None:
            return super().__str__()
        return self.name_metadata


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'extra-imports': [
            'pathlib',
            'typing',
            'numpy',
            'logger',
            'sklearn',
            'sklearn.decomposition',
            'sklearn.neighbors',
            'suffix_trees.STree',
        ],
        'allowed-io': ['WordEmbeddings.__init__'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200', 'E9998']
    })
