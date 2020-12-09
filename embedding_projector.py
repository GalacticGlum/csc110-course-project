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
        - weights_filepath: The path to the weights file.
        - vocab_filepath: The path to the vocab file.
        - name_metadata: The name of the word embeddings checkpoint.
    """
    # Private Instance Attributes:
    #   - _vocabulary: A dictionary mapping a word to its index.
    #   - _pca: Fitted sklearn PCA object.
    #   - _reduced_weights: Word embeddings reduced to a lower dimensional space.
    _vocabulary: Dict[str, int]
    _pca: decomposition.PCA
    _reduced_weights: np.ndarray

    def __init__(self, weights_filepath: Path, vocab_filepath: Path,
                 name_metadata: Optional[str] = None) -> None:
        """Initialize this word embeddings.

        Args:
            weights_filepath: Filepath to a numpy file containing trained model weights
                for the projection layer, corresponding to the learned embedding vectors
                of the words.
            vocab_filepath: A text file containing words of the vocabulary sorted in increasing
                order of the index, separated by new lines (i.e. the word on line 1 indicates
                the word with encoded index 0, and so on).
            name_metadata: The name of the word embeddings checkpoint.
        """
        with open(vocab_filepath) as file:
            self.words = file.read().splitlines()

        self.weights = np.load(weights_filepath)
        self.name_metadata = name_metadata
        self._vocabulary = {word: i for i, word in enumerate(self.words)}
        self._pca = None
        self._reduced_weights = None

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
        self._pca.fit(weights)
        self._reduced_weights = self._pca.transform(weights)
        return self._pca, self._reduced_weights

    def __getitem__(self, word: str) -> np.ndarray:
        """Return the embedding vector for the given word."""
        return self.get_vector(word)

    def __str__(self) -> str:
        """Return a string representation of this word embedding vector space."""
        if self.name_metadata is None:
            return super().__str__()
        return self.name_metadata


import plotly.express as px
import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


def _make_embedding_scatter(words: List[str], x: np.ndarray, y: np.ndarray, \
                            z: Optional[np.ndarray] = None) -> go.Figure:
    """Make a scatter plot given the embedding weight data.
    Return a 3D scatter plot if three dimensions were given, or a 2D scatter plot otherwise.

    Args:
        words: A list of strings containing the words of the model vocabulary.
        x: A numpy array containing the x-coordinates.
        y: A numpy array containing the y-coordinates.
        z: A numpy array containing the z-coordinates.
    """
    # Create the figure for displaying the embedding points
    if z is not None:
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
    else:
        embedding_fig = px.scatter(
            x=x, y=y,
            hover_name=words,
            opacity=0.5
        )

    return embedding_fig


def _make_app(embeddings_list: List[WordEmbeddings]) -> dash.Dash:
    """Make the Dash app for the embedding projector.

    Args:
        embeddings_list: A list of word embeddings.

    Preconditions:
        - len(embeddings_list) > 0
    """
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(children=[
        html.H1(children='Embedding Projector'),
        # We use a hidden div to run callbacks with no output element.
        # For example, when we want to do some data processing (such as computing the PCA).
        html.Div(id='hidden-div', style={'display': 'none'}),
        html.Div([
            html.Div([
                html.Div([
                    html.H3('DATA', style={'font-weight': 'bold'}),
                    html.Label('Embeddings'),
                    dcc.Dropdown(id='embeddings-dropdown',
                        options=[
                            {'label': str(embeddings), 'value': i}
                            for i, embeddings in enumerate(embeddings_list)
                        ],
                        clearable=False,
                        value=0
                    ),
                ]),
                html.Div([
                    html.H3('PCA', style={'font-weight': 'bold'}),
                    html.Label('X'),
                    dcc.Dropdown(id='x-component-dropdown', value=0),
                    html.Label('Y'),
                    dcc.Dropdown(id='y-component-dropdown', value=1),
                    html.Div([
                        html.Label('Z', className='ten columns'),
                        dcc.Checklist(id='use-z-component',
                            options=[{'label': '', 'value': 'yes'}],
                            value=['yes'],
                            className='two columns'
                        )
                    ], className='row'),
                    dcc.Dropdown(id='z-component-dropdown', value=2),
                    html.Br(),
                    html.Label(id='pca-variance-label')
                ])
            ], className='two columns'),
            html.Div([
                dcc.Loading(id='embedding-graph-loading',
                    children=[html.Div(id='embedding-graph')],
                    type='default'
                )
            ], className='ten columns')
        ], className='row'),
    ])

    @app.callback([
        Output('x-component-dropdown', 'options'),
        Output('y-component-dropdown', 'options'),
        Output('z-component-dropdown', 'options')],
        Input('embeddings-dropdown', 'value'))
    def embeddings_changed(index: int) -> Tuple[List[dict], List[dict], List[dict]]:
        """Triggered when the selected embedding changes.
        This function recomputes the PCA.

        Args:
            index: The index of the currently selected embeddings.
        """
        # Get embeddings
        embeddings = embeddings_list[index]
        pca, _ = embeddings.pca(force_rebuild=True)

        component_options = [
            {
                'label': f'Component #{i + 1} (var: {variance * 100:.2f}%)',
                'value': i
            }
            for i, variance in enumerate(pca.explained_variance_ratio_)
        ]

        return (
            component_options,  # Output for x-component-dropdown
            component_options,  # Output for y-component-dropdown
            component_options   # Output for z-component-dropdown
        )

    @app.callback([
        Output('embedding-graph', 'children'),
        Output('pca-variance-label', 'children')],
        [Input('embeddings-dropdown', 'value'),
        Input('x-component-dropdown', 'value'),
        Input('y-component-dropdown', 'value'),
        Input('z-component-dropdown', 'value'),
        Input('use-z-component', 'value')])
    def components_changed(index: int, x_component: int, y_component: int,
                           z_component: int, use_z_component_values: list) \
            -> Tuple[dash.Figure, str]:
        """Triggered when the PCA components are changed.
        Return the updated word embedding graph.

        Args:
            index: The index of the currently selected embeddings.
            x_component: The zero-based index of the PCA component to use for the X-axis.
            z_component: The zero-based index of the PCA component to use for the Y-axis.
            z_component: The zero-based index of the PCA component to use for the Z-axis.
            use_z_component_values: A list of values for the use_z_component
                checklist. Since this checklist contains a single element,
                the list is empty when the checklist is not toggled.
        """
        embeddings = embeddings_list[index]
        pca, weights = embeddings.pca()
        # Select the components
        components = [x_component, y_component]
        if use_z_component_values:
            components.append(z_component)

        weights = np.take(weights, components, axis=-1)

        # The indices where we want to split the weights matrix.
        # For example, if we have [1, 2, 3] and we want 3 vectors,
        # we would split the array at indices 1 and 2.
        split_indices = list(range(1, weights.shape[-1]))
        # Split the matrix into separate vectors containing components
        axes = np.squeeze(np.split(weights, split_indices, axis=1))
        # Update the embedding graph
        scatter = _make_embedding_scatter(embeddings.words, *axes)
        embedding_graph = dcc.Graph(
            figure=scatter,
            style={'height': '100vh'}
        )

        # Compute the total variance described the chosen components.
        # This is the sum of the variance described by each component.
        total_variance = np.sum(np.take(pca.explained_variance_ratio_, components))

        return (
            embedding_graph,
            f'Total variance described {total_variance * 100:.2f}%.'
        )

    @app.callback(
        Output('z-component-dropdown', 'disabled'),
        Input('use-z-component', 'value'))
    def toggle_use_z_component(use_z_component_values: list) -> bool:
        """
        Trigged when the use-z-component checklist is toggled.

        Args:
            use_z_component_values: A list of values for the use_z_component
                checklist. Since this checklist contains a single element,
                the list is empty when the checklist is not toggled.
        """
        return not bool(use_z_component_values)

    return app


def embedding_projector(embeddings_list: List[WordEmbeddings],
                        debug: Optional[bool] = False,
                        port: Optional[int] = 5006) -> None:
    """Start the embedding projector given word embeddings."""
    app = _make_app(embeddings_list)
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

    embeddings = WordEmbeddings(
        weights_filepath, vocab_filepath,
        name_metadata=weights_filepath.parent.stem
    )

    # Start the embedding projector
    embedding_projector([embeddings], debug=args.debug, port=args.port)


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
