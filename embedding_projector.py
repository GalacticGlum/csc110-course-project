"""Tool for visualising embeddings in 2D and 3D space."""

from __future__ import annotations

import heapq
import argparse
from typing import (
    Optional,
    Tuple,
    List,
    Dict
)

import numpy as np
from sklearn import decomposition
from pathlib import Path
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
    """
    # Private Instance Attributes:
    #   - _vocabulary: A dictionary mapping a word to its index.
    _vocabulary: Dict[str, int]

    def __init__(self, weights: np.ndarray, words: List[str]) -> None:
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
        self.weights = weights
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
            vector = self.weights[word_index]

        # A list of 2-element tuples containing the similarity to the search word
        # and the index of the word in the vocabulary.
        all_similarities = [
            (similarity_func(vector, u), i)
            for i, u in enumerate(self.weights)
            if i != word_index   # We don't want to include the search word
        ]

        # Use a min-heap to efficiently get most similar words
        most_similar = heapq.nlargest(k, all_similarities)
        most_similar = [
            (self.words[index], similarity)
            for similarity, index in most_similar
        ]

        return most_similar

    def get_vector(self, word: str) -> np.ndarray:
        """Return the embedding vector for the given word."""
        return self.weights[self._vocabulary[word]]

    def __getitem__(self, word: str) -> np.ndarray:
        """Return the embedding vector for the given word."""
        return self.get_vector(word)

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

import plotly.express as px
import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as html


def _make_embedding_scatter3d(weights: np.ndarray, words: List[str]) -> go.Figure:
    """Make a 3D scatter plot given the embedding weight data.

    Args:
        weights: 3-dimensional matrix of data containing the embeddings.
        words: A list of strings containing the words of the model vocabulary.
    """
    # Split the matrix into separate vectors containing components
    x, y, z = np.squeeze(np.split(weights, [1, 2], axis=1))

    # Create the figure for displaying the embedding points
    embedding_fig = px.scatter_3d(
        x=x, y=y, z=z,
        hover_name=words,
        opacity=0.5
    )

    # Update the axis settings
    axis_values = {
        'backgroundcolor': 'white',
        'gridcolor': '#D9D9D9'
    }
    embedding_fig.update_layout(
        scene={
            'xaxis': axis_values,
            'yaxis': axis_values,
            'zaxis': axis_values,
            'aspectmode': 'cube'
        }
    )

    return embedding_fig


def _make_app(reduced_embeddings: np.ndarray, words: List[str]) -> dash.Dash:
    """Make the Dash app for the embedding projector.

    Args:
        reduced_embeddings: The embedding vectors reduced into 3-dimensions.
    """
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(children=[
        html.H1(children='Embedding Projector'),
        html.Div([
            html.Div([
                html.Div([
                    html.H3('DATA', style={'font-weight': 'bold'}),
                ]),
                html.Div([
                    html.H3('PCA', style={'font-weight': 'bold'}),
                    html.Label('Component #1 (X)'),
                    dcc.Dropdown(
                        options=[
                            {'label': 'New York City', 'value': 'NYC'},
                            {'label': u'Montréal', 'value': 'MTL'},
                            {'label': 'San Francisco', 'value': 'SF'}
                        ],
                        value='NYC'
                    ),
                    html.Label('Component #1 (Y)'),
                    dcc.Dropdown(
                        options=[
                            {'label': 'New York City', 'value': 'NYC'},
                            {'label': u'Montréal', 'value': 'MTL'},
                            {'label': 'San Francisco', 'value': 'SF'}
                        ],
                        value='NYC'
                    ),
                    html.Div([
                        html.Label('Component #3 (Z)', className='ten columns'),
                        dcc.Checklist(
                            options=[{'label': '', 'value': 'use_z_component'}],
                            value=['use_z_component'],
                            className='two columns'
                        )
                    ], className='row'),
                    dcc.Dropdown(
                        options=[
                            {'label': 'New York City', 'value': 'NYC'},
                            {'label': u'Montréal', 'value': 'MTL'},
                            {'label': 'San Francisco', 'value': 'SF'}
                        ],
                        value='NYC'
                    )
                ])
            ], className='two columns'),
            html.Div([
                dcc.Graph(id='embedding-graph',
                          figure=_make_embedding_scatter3d(reduced_embeddings, words),
                          style={'height': '100vh'}
                )
            ], className='ten columns')
        ], className='row'),
    ])

    return app


def embedding_projector(embeddings: WordEmbeddings, debug: Optional[bool] = False,
                        port: Optional[int] = 5006) -> None:
    """Start the embedding projector given word embeddings."""
    logger.info('Spherizing data')
    # Shift each point by the centroid
    centroid = np.mean(embeddings.weights, axis=0)
    weights = embeddings.weights - centroid
    # Normalize data to unit vectors
    weights = weights / np.linalg.norm(weights, axis=-1, keepdims=True)

    logger.info('Computing PCA...')
    pca = decomposition.PCA(n_components=3)
    pca.fit(weights)
    reduced_weights = pca.transform(weights)

    app = _make_app(reduced_weights, embeddings.words)
    app.run_server(debug=debug, port=port)


def main(args: argparse.Namespace) -> None:
    """Main entrypoint for the script."""
    # Ensure that at least on data argument was provided
    if args.checkpoint_directory is None and \
       args.weights_filepath is None and \
        args.vocab_filepath is None:
        logger.error('One of --checkpoints / (--weights-filepath and --vocab-filepath) is required!')
        exit(1)

    if args.checkpoint_directory is not None:
        weights_filepath = args.checkpoint_directory / 'proj_weights.npy'
        vocab_filepath = args.checkpoint_directory / 'vocab.txt'
    else:
        weights_filepath = args.weights_filepath
        args.vocab_filepath = args.vocab_filepath

    embeddings = WordEmbeddings.load(weights_filepath, vocab_filepath)
    embedding_projector(embeddings, debug=args.debug, port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool for visualising embeddings in 2D and 3D space.')
    parser.add_argument('--checkpoint', dest='checkpoint_directory', type=Path, default=None,
                        help='Path to a checkpoint directory containing a numpy file with the trained '
                             'embedding weights (proj_weights.npy) and a text file with the model '
                             'vocabulary (vocab.txt)')
    parser.add_argument('-w', '--weights-filepath', type=Path, default=None,
                        help='Path to a numpy file containing the trained embedding weights. '
                             'Use this instead of specifying the checkpoint directory.')
    parser.add_argument('-v', '--vocab-filepath', type=Path, default=None,
                        help='Path to a text file containing the model vocabulary. '
                             'Use this instead of specifying the checkpoint directory.')
    parser.add_argument('--port', type=int, default=5006,
                        help='The port to open the server on. Defaults to 5006.')
    parser.add_argument('--debug', action='store_true', dest='debug',
                        help='Whether to run the app in debug mode.')
    parser.add_argument
    main(parser.parse_args())
