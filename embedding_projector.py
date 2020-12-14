"""Tool for visualising embeddings in 2D and 3D space."""

from __future__ import annotations

import json
import uuid
import argparse
from pathlib import Path
from typing import (
    Optional,
    Tuple,
    List
)

import numpy as np
import plotly.express as px
import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL

from logger import logger
from utils import rgb_lerp, rgb_to_str
from word_embeddings import WordEmbeddings


def _make_embedding_scatter(words: List[str], x: np.ndarray, y: np.ndarray,
                            z: Optional[np.ndarray] = None,
                            marker_colours: Optional[List[str]] = None,
                            marker_opacity: Optional[float] = 0.25,
                            marker_size: Optional[float] = 4.5) -> go.Figure:
    """Make a scatter plot given the embedding weight data.
    Return a 3D scatter plot if three dimensions were given, or a 2D scatter plot otherwise.

    Args:
        words: A list of strings containing the words of the model vocabulary.
        x: A numpy array containing the x-coordinates.
        y: A numpy array containing the y-coordinates.
        z: A numpy array containing the z-coordinates.
        marker_colours: The colours of the points. The i-th element of this list corresponds
            to the colour of the market with coordinates (x[i], y[i], z[i]).
        marker_opacity: The opacity of the markers.
        marker_size: The size of the markers.
    Preconditions:
        - len(x) == len(y)
        - z is None or len(x) == len(z)
        - marker_colours is None or len(x) == len(marker_colours)
    """
    DEFAULT_MARKER_COLOUR = 'rgb(1, 135, 75)'
    if marker_colours is None:
        marker_colours = [DEFAULT_MARKER_COLOUR] * len(words)

    # Create the figure for displaying the embedding points
    if z is not None:
        embedding_fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker={
                'size': marker_size,
                'opacity': marker_opacity,
                'color': marker_colours
            },
            hovertemplate='<b>%{text}</b><br><br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>',
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
                                 value=0),
                    html.Div(id='metadata-div'),
                    dbc.Checklist(id='spherize-data',
                                  options=[{'label': 'Spherize data', 'value': 'yes'}],
                                  value=['yes'],
                                  inline=True,
                                  switch=True)
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
                                  className='py-1'),
                    dcc.Dropdown(id='z-component-dropdown', value=2, clearable=False),
                    html.Br(),
                    html.Label(id='pca-variance-label')
                ])
            ]), width=2),
            dbc.Col(html.Div([
                dcc.Loading(id='embedding-graph-loading',
                            children=[
                                dcc.Graph(
                                    # Make a defualt empty graph
                                    figure=_make_embedding_scatter([], [], [], []),
                                    id='embedding-graph',
                                    style={'height': '100vh'}
                                )
                            ],
                            type='default')
            ]), width=8),
            dbc.Col(html.Div([
                html.Div([
                    html.H3('ANALYSIS', style={'font-weight': 'bold'}),
                    dbc.Row([
                        dbc.Col([dbc.Button('Show All Data',
                                            id='show-all-data-button',
                                            outline=True,
                                            color='dark',
                                            className='mr-2 my-4',
                                            disabled=True)]),
                        dbc.Col([dbc.Button('Isolate Points',
                                            id='isolate-points-button',
                                            outline=True,
                                            color='dark',
                                            className='mr-2 my-4',
                                            disabled=True)]),
                        dbc.Col([dbc.Button('Clear Selection',
                                            id='clear-selection-button',
                                            outline=True,
                                            color='dark',
                                            className='my-4',
                                            disabled=True)])
                    ], no_gutters=True),
                    html.Div(id='selected-word-state', style={'display': 'none'}),
                    dbc.Tabs([
                        dbc.Tab(tab_id='search-tab', children=[
                            html.Div([
                                dbc.Input(id='word-search-input',
                                          type='text',
                                          placeholder='Search'),
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
                        ),
                    ], id='analysis-tabs', active_tab='search-tab')
                ]),
            ]), width=2)
        ])
    ])


def _make_callbacks(app: dash.Dash, embeddings_list: List[WordEmbeddings]) -> None:
    """Make the callbacks for the embedding projector app."""
    @app.callback([Output('x-component-dropdown', 'options'),
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

    @app.callback([Output('embedding-graph', 'figure'),
                   Output('pca-variance-label', 'children'),
                   Output('show-all-data-button', 'disabled')],
                  [Input('x-component-dropdown', 'value'),
                   Input('y-component-dropdown', 'value'),
                   Input('z-component-dropdown', 'value'),
                   Input('use-z-component', 'value'),
                   Input('update-embeddings-signal', 'children'),
                   Input('isolate-points-button', 'n_clicks'),
                   Input('clear-selection-button', 'n_clicks'),
                   Input('show-all-data-button', 'n_clicks')],
                  [State('embeddings-dropdown', 'value'),
                   State('embedding-graph', 'figure'),
                   State('pca-variance-label', 'children'),
                   State('selected-word-state', 'children'),
                   State('show-all-data-button', 'disabled')])
    def components_changed(x_component: int, y_component: int, z_component: int,
                           use_z_component_values: list, signal: object,
                           isolate_points_n_clicks: int, clear_search_selection_n_clicks: int,
                           show_all_data_button_n_clicks: int, index: int,
                           current_embedding_figure: dash.Figuure, pca_variance_label: str,
                           selected_word: str, show_all_data_button_disabled: bool) \
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
            isolate_points_n_clicks: The number of times the isolate points button has
                been clicked.
            clear_search_selection_n_clicks: The number of times the clear search selection
                button has been clicked.
            show_all_data_button_n_clicks: The number of times the show all data button has
                been clicked.
            index: The index of the currently selected embeddings.
            current_embedding_figure: The current embedding figure.
            pca_variance_label: The current PCA variance label.
            selected_word: The currently selected word.
        """
        # Load the embeddings and PCA reduced weights.
        embeddings = embeddings_list[index]
        pca, weights = embeddings.pca()
        words = embeddings.words

        # Marker settings
        marker_colours = None
        marker_opacity = 0.25
        marker_size = 4.5

        # Get the callback context (from where was it called?)
        ctx = dash.callback_context
        triggered = ctx.triggered[0]
        # If the triggered prop_id contains 'n_clicks' then it was triggered
        # by a search result list item. Otherwise, it was triggered by clicking on the graph.
        if triggered['prop_id'] == 'isolate-points-button.n_clicks':
            # Isolate the selected word by finding its neighbours...
            similarities = embeddings.most_similar(selected_word, k=100)
            # Add the original word to the list so that it is included in the plot.
            similarities += [(selected_word, 1.0)]
            words = [word for word, _ in similarities]

            # Take only the weights of the neighbours...
            weights = np.array([embeddings.get_vector(word) for word in words])
            # Enable the show all data button
            show_all_data_button_disabled = False

            # Override the colours on the plot.
            MOST_DIFFERENT_COLOUR = (255, 252, 0)
            MOST_SIMILAR_COLOUR = (255, 0, 0)

            marker_colours = [
                rgb_to_str(rgb_lerp(MOST_DIFFERENT_COLOUR, MOST_SIMILAR_COLOUR, similarity))
                for _, similarity in similarities
            ]

            marker_opacity = 0.6
            marker_size = 10
        elif triggered['prop_id'] == 'show-all-data-button.n_clicks':
            show_all_data_button_disabled = not show_all_data_button_disabled

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
        scatter = _make_embedding_scatter(
            words, *axes,
            marker_colours=marker_colours,
            marker_opacity=marker_opacity,
            marker_size=marker_size
        )

        # Changing the value of uirevision causes the graph to update!
        # So, let's initialise it to some unique value. This will ensure
        # that Dash updates the user state for the graph.
        scatter.update_layout(uirevision=str(uuid.uuid4()))

        # Compute the total variance described the chosen components.
        # This is the sum of the variance described by each component.
        total_variance = np.sum(np.take(pca.explained_variance_ratio_, components))
        label = f'Total variance described: {total_variance * 100:.2f}%.'

        return (scatter, label, show_all_data_button_disabled)

    @app.callback(Output('z-component-dropdown', 'disabled'),
                  Input('use-z-component', 'value'))
    def toggle_use_z_component(use_z_component_values: list) -> bool:
        """Trigged when the use-z-component checklist is toggled.

        Args:
            use_z_component_values: A list of values for the use_z_component
                checklist. Since this checklist contains a single element,
                the list is empty when the checklist is not toggled.
        """
        return not bool(use_z_component_values)

    @app.callback([Output('word-search-results', 'children'),
                   Output('word-search-matches', 'children')],
                  [Input('word-search-input', 'value'),
                   Input('clear-selection-button', 'n_clicks')],
                  State('embeddings-dropdown', 'value'))
    def on_search_changed(search_term: str, n_clicks: int, index: int) -> Tuple[List[dbc.ListGroupItem], str]:
        """Triggered when the search box changes."""
        # The number of search results to show.
        SHOW_FIRST_N = 100

        ctx = dash.callback_context
        clear_selection_triggered = n_clicks is not None and n_clicks > 0 and \
            ctx.triggered[0]['prop_id'] == 'clear-selection-button.n_clicks'
        if index is None or not search_term or clear_selection_triggered:
            return ([], '')

        embeddings = embeddings_list[index]
        search_results = embeddings.search_words(search_term)

        results = []
        for word in search_results[:SHOW_FIRST_N]:
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

    @app.callback(Output('word-search-input', 'value'),
                  Input('clear-selection-button', 'n_clicks'))
    def clear_search_selection(n_clicks: int) -> str:
        """Triggered when the clear search selection button is clicked.
        Args:
            n_clicks: The number of times the button was clicked.
        """
        return ''

    @app.callback([Output('selected-word-tab', 'children'),
                   Output('selected-word-tab', 'disabled'),
                   Output('analysis-tabs', 'active_tab'),
                   Output('clear-selection-button', 'disabled'),
                   Output('isolate-points-button', 'disabled'),
                   Output('selected-word-state', 'children')],
                  [Input('embedding-graph', 'clickData'),
                   Input({'type': 'search-result', 'index': ALL, 'word': ALL}, 'n_clicks'),
                   Input('clear-selection-button', 'n_clicks')],
                  State('embeddings-dropdown', 'value'))
    def update_word_selection(click_data: dict, n_clicks: List[int], clear_selection_n_clicks: int,
                              index: int) -> None:
        """Triggered when a new word is selected.

        Args:
            n_clicks: The number of times each search result list item has been clicked.
            click_data: Information about the point the user clicked on the embedding graph.
            clear_selection_n_clicks: The number of times the clear search selection button
                has been clicked.
            index: The index of the currently selected word embeddigns object.
        """
        # The default/empty return value.
        DEFAULT = ([], True, 'search-tab', True, True, '')

        ctx = dash.callback_context
        # If the context is None, then we can't trace the event, so return.
        if ctx is None:
            return DEFAULT

        triggered = ctx.triggered[0]

        # if the user clicked on the clear search selection button, then we should clear the tab.
        clear_selection_triggered = clear_selection_n_clicks is not None \
            and clear_selection_n_clicks > 0 \
            and ctx.triggered[0]['prop_id'] == 'clear-selection-button.n_clicks'

        if clear_selection_triggered:
            return DEFAULT

        # If the triggered prop_id contains 'n_clicks' then it was triggered
        # by a search result list item. Otherwise, it was triggered by clicking on the graph.
        if 'n_clicks' in triggered['prop_id']:
            if n_clicks is None:
                return DEFAULT

            # Check if the click counts are all zero..
            # In which case, the user hasn't clicked on a button.
            not_has_clicked = all(x == 0 for x in n_clicks)
            # If the event was triggered from multiple sources, then we know that
            # this wasn't a click event (rather, an init event), since the user can't
            # click on multiple buttons at the same time.
            if not_has_clicked or len(ctx.triggered) > 1:
                return DEFAULT

            # The prop_id is a string of the form '{index data}.n_clicks' where index data
            # is a JSON-like string containing information about the element ID.
            # This can contain custom data, and in our case, we store the word of each element here.
            id_dict = json.loads(triggered['prop_id'].split('.')[0])
            selected_word = id_dict['word']
        elif 'clickData' in triggered['prop_id']:
            # Make sure that the click data is not None or empty
            if click_data is None or len(click_data) == 0:
                return DEFAULT

            point_data = click_data['points'][0]
            # Depending on the type of plot, the selected word is either contained in
            # the text or hovertext attribute. Get whichever is not None.
            selected_word = point_data.get('text', None) or point_data.get('hovertext', None)
        else:
            return DEFAULT

        # Get most similar words in the embedding space
        embeddings = embeddings_list[index]
        most_similar = embeddings.most_similar(selected_word, k=100)

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
            dbc.FormText(selected_word, className='pt-3'),
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

        return (tab_contents, False, 'selected-word-tab', False, False, selected_word)


def _make_app(embeddings_list: List[WordEmbeddings]) -> dash.Dash:
    """Make the Dash app for the embedding projector.

    Args:
        embeddings_list: A list of word embeddings.

    Preconditions:
        - len(embeddings_list) > 0
    """
    external_stylesheets = [dbc.themes.BOOTSTRAP]
    app = dash.Dash(
        __name__, external_stylesheets=external_stylesheets,
        title='Embedding Projector',
        update_title=None
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
    if args.checkpoint_directory is None \
       and args.weights_filepath is None \
       and args.vocab_filepath is None:

        logger.error('One of --checkpoints / (--weights-filepath '
                     'and --vocab-filepath) is required!')
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
    # import python_ta
    # python_ta.check_all(config={
    #     'extra-imports': [
    #         'json',
    #         'uuid',
    #         'argparse',
    #         'pathlib',
    #         'typing',
    #         'numpy',
    #         'logger',
    #         'utils',
    #         'word_embeddings',
    #         'plotly.express',
    #         'plotly.graph_objs',
    #         'dash',
    #         'dash_core_components',
    #         'dash_html_components',
    #         'dash_bootstrap_components',
    #         'dash.dependencies'
    #     ],
    #     'allowed-io': [''],
    #     'max-line-length': 100,
    #     'disable': ['R1705', 'C0200', 'W0612']
    # })

    parser = argparse.ArgumentParser(description='Tool for visualising '
                                                 'embeddings in 2D and 3D space.')
    parser.add_argument('--checkpoint', dest='checkpoint_directory', type=Path, default=None,
                        help='Path to a checkpoint directory containing a numpy file with '
                             'the trained embedding weights (proj_weights.npy) and a text '
                             'file with the model vocabulary (vocab.txt)')
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
    main(parser.parse_args())
