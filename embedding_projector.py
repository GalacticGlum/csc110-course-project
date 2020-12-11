"""Tool for visualising embeddings in 2D and 3D space."""

from __future__ import annotations

import json
import uuid
import argparse
from pathlib import Path
from typing import (
    Optional,
    Tuple,
    List,
    Dict,
    Generator
)

import numpy as np
from logger import logger
from utils import rgb_lerp, rgb_to_str
from sklearn import decomposition, neighbors
from suffix_trees.STree import STree as SuffixTree

import plotly.express as px
import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL


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
    #   - _all_words: A space-separated string containing all the words in the vocabulary.
    #   - _suffix_tree: A suffix tree for fast searching of the vocabulary.
    #   - _nearest_neighbours: A nearest neighbours model for finding most similar embeddings.
    _vocabulary: Dict[str, int]
    _pca: decomposition.PCA
    _reduced_weights: np.ndarray
    _all_words: str
    _suffix_tree: SuffixTree
    _nearest_neighbours: neighbors.NearestNeighbors

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

        self._build_suffix_tree()
        self._build_nearest_neighbours()

    def _build_suffix_tree(self) -> None:
        """Build a suffix tree from the vocabulary."""
        logger.info('Building suffix tree!')
        self._all_words = ' '.join(self.words)
        self._suffix_tree = SuffixTree(self._all_words)
        logger.info('Finished building suffix tree!')

    def _build_nearest_neighbours(self) -> None:
        """Build a nearest neighbour searcher from the embedding vectors."""
        logger.info('Building nearest neighbours!')

        # We use a KNN model to perform embedding similarity search quickly.
        # The goal is to find the most similar embedding vectors based on their cosine similarity.
        # However, while KNN does not support the cosine metric, by normalizing the embedding vectors,
        # we can use a KNN on Euclidean distance to find the most similar vectors, and we will get the
        # same ordering as we would if we used cosine similarity.
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
        distances, indices = self._nearest_neighbours.kneighbors([vector], n_neighbors=k)

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
        embedding_fig = go.FigureWidget(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker={
                'size': 4.5,
                'opacity': 0.25,
                'color': ['rgb(1, 135, 75)'] * len(words)
            },
            hovertemplate=
            '<b>%{text}</b><br>' +
            '<br>x: %{x}' +
            '<br>y: %{y}' +
            '<br>z: %{z}' +
            '<extra></extra>',
            text=words
        )])

        # Update the axis settings
        axis_values = {
            'backgroundcolor': 'white',
            'gridcolor': 'rgb(217, 217, 217)'
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

    # Update common layout options
    embedding_fig.update_layout(
        hoverlabel={
            'bgcolor': 'white',
            'font_size': 16,
            'font_family': 'Arial'
        }
    )

    return embedding_fig

def _make_layout(embeddings_list: List[WordEmbeddings]) -> object:
    """Create the layout for the embedding projector app."""
    return dbc.Container(fluid=True, children=[
        html.H1(children='Embedding Projector'),
        # We use a hidden div to faciliate callbacks.
        # This 'signal' can be used both ways: to trigger a callback, or to have a callback
        # return nothing.
        #
        # For example, when we want to do some data processing (such as computing the PCA)
        # but may not necessarily want to render anything new.
        html.Div(id='update-embeddings-signal', style={'display': 'none'}),
        html.Div(id='hidden-div', style={'display': 'none'}),
        dcc.Loading(id='loading-spinner', fullscreen=True, loading_state={'is_loading': True}),
        dbc.Row([
            dbc.Col(html.Div([
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
                    html.Div(id='metadata-div'),
                    dbc.Checklist(id='spherize-data',
                        options=[{'label': 'Spherize data', 'value': 'yes'}],
                        value=['yes'],
                        inline=True,
                        switch=True
                    )
                ]),
                html.Div([
                    html.H3('PCA', style={'font-weight': 'bold'}),
                    dbc.Label('X-axis'),
                    dcc.Dropdown(id='x-component-dropdown', value=0, clearable=False),
                    dbc.Label('Y-axis'),
                    dcc.Dropdown(id='y-component-dropdown', value=1, clearable=False),
                    dbc.Checklist(id='use-z-component',
                        options=[{'label': 'Z-Axis', 'value': 'yes'}],
                        value=['yes'],
                        inline=True,
                        switch=True,
                        className='py-1'
                    ),
                    dcc.Dropdown(id='z-component-dropdown', value=2, clearable=False),
                    html.Br(),
                    html.Label(id='pca-variance-label')
                ])
            ]), width=2),
            dbc.Col(html.Div([
                dcc.Graph(
                    # Make a defualt empty graph
                    figure=_make_embedding_scatter([], [], [], []),
                    id='embedding-graph',
                    style={'height': '100vh'}
                )
            ]), width=8),
            dbc.Col(html.Div([
                html.Div([
                    html.H3('ANALYSIS', style={'font-weight': 'bold'}),
                    dbc.Row([
                        dbc.Col([dbc.Button('Show All Data',
                            outline=True,
                            color='dark',
                            className='mr-2 my-4',
                            disabled=True
                        )]),
                        dbc.Col([dbc.Button('Isolate Points',
                            outline=True,
                            color='dark',
                            className='mr-2 my-4',
                            disabled=True
                        )]),
                        dbc.Col([dbc.Button('Clear Selection',
                            outline=True,
                            color='dark',
                            className='my-4',
                            disabled=True
                        )])
                    ], no_gutters=True),
                    dbc.Tabs([
                        dbc.Tab(tab_id='search-tab', children=[
                            html.Div([
                                dbc.Input(id='word-search-input',
                                    type='text',
                                    placeholder='Search'
                                ),
                                dbc.FormText(id='word-search-matches', color='secondary'),
                                html.Div([
                                    dbc.ListGroup(id='word-search-results', className='pt-3')
                                ], className='overflow-auto', style={
                                    'max-height': '50vh',
                                    'height': '100%'
                                })
                            ], className='mt-3')
                        ], label='Search'),
                        dbc.Tab(
                            id='selected-word-tab',
                            tab_id='selected-word-tab',
                            label='Selection',
                            disabled=True
                        )
                    ], id='analysis-tabs', active_tab='search-tab')
                ]),
            ]), width=2)
        ])
    ])


def _make_callbacks(app: dash.Dash, embeddings_list: List[WordEmbeddings]) -> None:
    """Make the callbacks for the embedding projector app."""
    @app.callback([
        Output('x-component-dropdown', 'options'),
        Output('y-component-dropdown', 'options'),
        Output('z-component-dropdown', 'options'),
        Output('metadata-div', 'children'),
        Output('update-embeddings-signal', 'children')],
        [Input('embeddings-dropdown', 'value'),
        Input('spherize-data', 'value')])
    def embeddings_changed(index: int, spherize_data_values: list) \
            -> Tuple[List[dict], List[dict], List[dict], str]:
        """Triggered when the selected embedding changes.
        This function recomputes the PCA, and triggers the components_changed callback.

        Args:
            index: The index of the currently selected embeddings.
            spherize_data_values: A list of values for the spherize_data checklist.
                Since this checklist contains a single element, the list is empty
                when the checklist is not toggled, and can be treated like a bool.
        """
        # Get embeddings
        embeddings = embeddings_list[index]
        pca, _ = embeddings.pca(spherize=bool(spherize_data_values), force_rebuild=True)

        component_options = [
            {
                'label': f'Component #{i + 1} (var: {variance * 100:.2f}%)',
                'value': i
            }
            for i, variance in enumerate(pca.explained_variance_ratio_)
        ]

        metadata_div = [
            html.Br(),
            html.Div([html.P(f'Points: {len(embeddings.weights)}')]),
            html.Div([html.P(f'Dimensions: {embeddings.weights.shape[-1]}')])
        ]

        return (
            component_options,  # Output for x-component-dropdown
            component_options,  # Output for y-component-dropdown
            component_options,  # Output for z-component-dropdown,
            metadata_div,       # Output for metadata-div
            # We output a dummy value for the signal, but that is unique
            # (so that any callback that uses this signal is trigged).
            str(uuid.uuid4()),  # Output for the signal div.
        )

    @app.callback([
        Output('embedding-graph', 'figure'),
        Output('pca-variance-label', 'children')],
        [Input('x-component-dropdown', 'value'),
        Input('y-component-dropdown', 'value'),
        Input('z-component-dropdown', 'value'),
        Input('use-z-component', 'value'),
        Input('update-embeddings-signal', 'children')],
        State('embeddings-dropdown', 'value'))
    def components_changed(x_component: int, y_component: int, z_component: int,
                           use_z_component_values: list, signal: object, index: int) \
            -> Tuple[dash.Figure, str]:
        """Triggered when the PCA components are changed.
        Return the updated word embedding graph.

        Args:
            x_component: The zero-based index of the PCA component to use for the X-axis.
            z_component: The zero-based index of the PCA component to use for the Y-axis.
            z_component: The zero-based index of the PCA component to use for the Z-axis.
            use_z_component_values: A list of values for the use_z_component
                checklist. Since this checklist contains a single element,
                the list is empty when the checklist is not toggled.
            index: The index of the currently selected embeddings.
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

        # Compute the total variance described the chosen components.
        # This is the sum of the variance described by each component.
        total_variance = np.sum(np.take(pca.explained_variance_ratio_, components))

        return (
            scatter,
            f'Total variance described: {total_variance * 100:.2f}%.'
        )

    @app.callback(
        Output('z-component-dropdown', 'disabled'),
        Input('use-z-component', 'value'))
    def toggle_use_z_component(use_z_component_values: list) -> bool:
        """Trigged when the use-z-component checklist is toggled.

        Args:
            use_z_component_values: A list of values for the use_z_component
                checklist. Since this checklist contains a single element,
                the list is empty when the checklist is not toggled.
        """
        return not bool(use_z_component_values)

    # @app.callback(
    #     Output('hidden-div', 'children'),
    #     Input('embedding-graph', 'clickData'))
    # def on_click_embedding_graph(click_data: dict) -> None:
    #     """Trigged when a point is clicked on the embedding graph.

    #     Args:
    #         click_data: A dictionary containing the information of the points
    #             the user is clicked on.
    #     """
    #     # print(click_data)

    @app.callback([
        Output('word-search-results', 'children'),
        Output('word-search-matches', 'children')],
        Input('word-search-input', 'value'),
        State('embeddings-dropdown', 'value'))
    def on_search_changed(search_term: str, index: int) -> Tuple[List[dbc.ListGroupItem], str]:
        """Triggered when the search box changes."""
        if index is None or not search_term:
            return ([], '')

        embeddings = embeddings_list[index]
        search_results = embeddings.search_words(search_term)

        results = []
        for word in search_results[:25]:
            # Create a new element for the result element, with the "search-result" type.
            # The element has a unique id so that the events don't clash.
            element_id = {
                'type': 'search-result',
                'index': str(uuid.uuid4()),
                'word': word
            }

            # Create the list group item element
            results.append(dbc.ListGroupItem(
                word, id=element_id,
                n_clicks=0, action=True
            ))

        return (
            results,
            # Create the label showing the number of matches found.
            'Found {} matches{}.'.format(
                len(search_results),
                f' (showing first {len(results)})' if len(search_results) > len(results) else ''
            )
        )

    @app.callback(
        [Output('selected-word-tab', 'children'),
        Output('selected-word-tab', 'disabled'),
        Output('analysis-tabs', 'active_tab')],
        Input({'type': 'search-result', 'index': ALL, 'word': ALL}, 'n_clicks'),
        State('embeddings-dropdown', 'value'))
    def on_search_result_clicked(n_clicks: List[int], index: int) -> None:
        """Triggered when any search result is clicked."""
        ctx = dash.callback_context
        # If the context is None, then we can't trace the event, so return.
        #
        # Or, if the event was triggered from multiple sources, then we know that
        # this wasn't a click event (rather, an init event), since the user can't
        # click on multiple buttons at the same time.
        not_has_clicked = all(x == 0 for x in n_clicks)
        if not_has_clicked or ctx is None or len(ctx.triggered) > 1:
            return ([], True, 'search-tab')

        triggered = ctx.triggered[0]
        if triggered.get('value') is None:
            return ([], True, 'search-tab')

        # The prop_id is a string of the form '{index data}.n_clicks' where index data
        # is a JSON-like string containing information about the element ID.
        # This can contain custom data, and in our case, we store the word of each element here.
        id_dict = json.loads(triggered['prop_id'].split('.')[0])

        # Get most similar words in the embedding space
        embeddings = embeddings_list[index]
        most_similar = embeddings.most_similar(id_dict['word'], k=100)

        # The colours for a similarity score of 0 and 1 respectively.
        MIN_SCORE_COLOUR = (133, 100, 4)
        MAX_SCORE_COLOUR = (21, 87, 36)

        # Build the contents of the selection tab.
        elements = []
        for word, similarity in most_similar:
            # A colour based on how strong the similarity score is.
            label_colour = rgb_to_str(rgb_lerp(MIN_SCORE_COLOUR, MAX_SCORE_COLOUR, similarity))
            elements.append(dbc.ListGroupItem(
                html.Div(
                    children=[
                        html.Div(word),
                        html.Div(
                            html.B(f'{similarity:.3f}'),
                            style={'color': label_colour}
                        )
                    ],
                    className='d-flex justify-content-between'
                )
            ))

        tab_contents = html.Div([
            dbc.FormText(id_dict['word'], className='pt-3'),
            html.Label('Nearest points in vector space:'),
            dbc.ListGroup(
                elements,
                className='overflow-auto',
                style={
                    'max-height': '50vh',
                    'height': '100%'
                }
            )
        ])

        return (tab_contents, False, 'selected-word-tab')


def _make_app(embeddings_list: List[WordEmbeddings]) -> dash.Dash:
    """Make the Dash app for the embedding projector.

    Args:
        embeddings_list: A list of word embeddings.

    Preconditions:
        - len(embeddings_list) > 0
    """
    # external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    external_stylesheets = [dbc.themes.BOOTSTRAP]
    app = dash.Dash(
        __name__, external_stylesheets=external_stylesheets,
        title='Embedding Projector'
    )

    # Setup the app
    app.layout = _make_layout(embeddings_list)
    _make_callbacks(app, embeddings_list)

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
