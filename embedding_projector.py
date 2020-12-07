"""Tool for visualising embeddings in 2D and 3D space."""

from __future__ import annotations

import heapq
import numpy as np
from pathlib import Path
from typing import (
    Optional,
    Tuple,
    List,
    Dict
)


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
        - embeddings: A matrix with shape (vocab_size, n) where n is the dimensionality
            of the embedding vectors (i.e. the number of components). The i-th row of
            the matrix should corresponding to the embedding vector for the word with
            encoded index i.
        - words: A list of strings, where the i-th element of the list
                corresponds to the word with encoded index i.
    """
    # Private Instance Attributes:
    #   - _vocabulary: A dictionary mapping a word to its index.
    _vocabulary: Dict[str, int]

    def __init__(self, embeddings: np.ndarray, words: List[str]) -> None:
        """Initialize this word embeddings.

        Args:
            embeddings: A matrix with shape (vocab_size, n) where n is the dimensionality
                of the embedding vectors (i.e. the number of components). The i-th row of
                the matrix should corresponding to the embedding vector for the word with
                encoded index i.

                This is also the size of the hidden layer in the Word2Vec model.
            words: A list of strings, where the i-th element of the list
                corresponds to the word with encoded index i.
        """
        self.embeddings = embeddings
        self.words = words
        self._vocabulary = {word: i for i, word in enumerate(words)}

    def most_similar(self, word: Optional[str] = None, k: Optional[int] = None,
                     similarity_func: Optional[callable] = None,
                     vector: Optional[np.ndarray] = None) \
            -> List[Tuple[str, float]]:
        """Finds the most common words to the given word.

        Return a list of 2-element tuple of the word and similarity,
        sorted in decreasing order of the similarity.

        If the given word is not in the vocabulary, an empty list is returned.

        Args:
            word: The search word. Required if vector is not specified.
            k: The number of most similar words to return.
                If unspecified, all words in the vocabulary are returned.
            similarity_func: A function which takes two embedding vectors
                (represented as one-dimensional numpy arrays) as input and
                returns a float indicating how similar the two vectors.

                If unspecified, defaults to the cosine similarity metric.
            vector: A vector with the same number of dimensions as the vector
                embeddings to search instead of a word.
        """
        assert word is not None or vector is not None
        if word is not None and word not in self._vocabulary:
            return []

        # Default to the vocab size
        # Clamp the given value of k to be in the range [0, vocab_size].
        vocab_size = len(self.words)
        k = max(min(k or vocab_size, vocab_size), 0)

        # Default to the cosine similarity metric
        similarity_func = similarity_func or cosine_similarity

        word_index = None
        if vector is None:
            word_index = self._vocabulary[word]
            vector = self.embeddings[word_index]

        # A list of 2-element tuples containing the similarity to the search word
        # and the index of the word in the vocabulary.
        all_similarities = [
            (similarity_func(vector, u), i)
            for i, u in enumerate(self.embeddings)
            if i != word_index   # We don't want to include the search word
        ]

        # Use a min-heap to efficiently get most similar words
        most_similar = heapq.nlargest(k, all_similarities)
        most_similar = [
            (self.words[index], similarity)
            for similarity, index in most_similar
        ]

        return most_similar

    def get_embedding(self, word: str) -> np.ndarray:
        """Return the embedding vector for the given word."""
        return self.embeddings[self._vocabulary[word]]

    def __getitem__(self, word: str) -> np.ndarray:
        """Return the embedding vector for the given word."""
        return self.get_embedding(word)

    @classmethod
    def load(cls, weights_filepath: Path, vocab_filepath: Path) -> WordEmbeddings:
        """Load embeddings from file.

        Args:
            weights_filepath: Filepath to a numpy file containing trained model weights
                for the projection layer, corresponding to the learned embedding vectors
                of the words.
            vocab_filepath: A text file containing words of the vocabulary sorted in increasing
                order of the index, separated by new lines (i.e. the word on line 1 indicates
                the word with encoded index 0, and so on).
        """
        with open(vocab_filepath) as file:
            words = file.read().splitlines()
        embeddings = np.load(weights_filepath)
        return cls(embeddings, words)