"""A tool for visualising k-hop graphs."""


import tqdm
import logging
import argparse
from pathlib import Path
from typing import Set, Optional

import numpy as np
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import community as community_louvain
from matplotlib.collections import LineCollection

from fa2 import ForceAtlas2
from sklearn import metrics
from curved_edges import curved_edges

from logger import logger
from word_embeddings import WordEmbeddings


def build_infinity_hop_graph(embeddings: WordEmbeddings, alpha: Optional[float] = 0.50) \
        -> nx.Graph:
    """Builds the infinity-hop graph for a word embeddings space.

    Args:
        embeddings: The word embeddings to generate the graph for.
        alpha: The similarity threshold. Words that have a cosine similarity
            of at least this threshold are kept, and the rest are discarded.
    """
    graph = nx.Graph()
    logger.info('Generating infinity-hop graph')
    weights = embeddings.weights.astype(np.float32)
    # Compute the cosine similarity between all pairs of embedding vector.
    similarities = metrics.pairwise.cosine_similarity(embeddings.weights)
    # Filter out similarity scores that are less than the threshold
    pairs = np.argwhere(similarities >= alpha)

    # Generate the infinity-hop network
    for pair in tqdm.tqdm(pairs):
        i, j = pair
        if i == j: continue
        # The weight of the edge is the cosine similary.
        graph.add_edge(i, j, weight=similarities[i][j])

    return graph


def build_k_hop_graph(embeddings: WordEmbeddings, target_word: str,
                      k: int, alpha: Optional[float] = 0.50) -> nx.Graph:
    """Builds the k-hop graph for a word embeddings space.

    Args:
        embeddings: The word embeddings to generate the graph for.
        target_word: The word of interest.
        k: The number of 'hops' between the word of interest and every node
            in the graph. The resultant graph has the property that the word
            of interest is reachable from any node in at most k edges.
        alpha: The similarity threshold. Words that have a cosine similarity
            of at least this threshold are kept, and the rest are discarded.
    """
    graph = build_infinity_hop_graph(embeddings, alpha)

    # Get the word index of the word of interest.
    T = embeddings._vocabulary[target_word]

    # Compute the shortest paths from the word of interest to all reachable nodes.
    logger.info('Computing shortest paths')
    paths = nx.single_source_shortest_path_length(graph, T)

    logger.info('Building k-hop graph')
    nodes_to_delete = set()
    for node in tqdm.tqdm(graph.nodes):
        # Remove the node if the word of interest is not reachable in at most k edges.
        if node not in paths or paths[node] > k:
            nodes_to_delete.add(node)

    for node in nodes_to_delete:
        graph.remove_node(node)

    logger.info('Generated k-hop graph (nodes: {}, edges: {})'.format(
        len(graph.nodes), len(graph.edges)
    ))
    return graph


def draw_k_hop_graph(embeddings: WordEmbeddings, target_word: str,
                     k: int, alpha: Optional[float] = 0.50,
                     min_node_size: Optional[float] = 20,
                     max_node_size: Optional[float] = 120,
                     min_font_size: Optional[float] = 6,
                     max_font_size: Optional[float] = 24) -> None:
    """Draw the k-hop graph for the given word embeddings and interest word.
    This function DOES NOT show the matplotlib plot.

    Args:
        embeddings: The word embeddings to generate the graph for.
        target_word: The word of interest.
        k: The number of 'hops' between the word of interest and every node
            in the graph. The resultant graph has the property that the word
            of interest is reachable from any node in at most k edges.
        alpha: The similarity threshold. Words that have a cosine similarity
            of at least this threshold are kept, and the rest are discarded.
        min_node_size: The minimum size of a node, in pixels.
        max_node_size: The maximum size of a node, in pixels.
        min_font_size: The minimum size of a label, in pixels.
        max_font_size: The maximum size of a label, in pixels.
    """
    if alpha is None:
        _, similarity  = embeddings.most_similar(target_word, k=1)[0]
        alpha = similarity - 0.05

    graph = build_k_hop_graph(embeddings, target_word, k, alpha=alpha)

    logger.info('Computing best partition (Louvain community detection)')
    # compute the best partition
    partition = community_louvain.best_partition(graph)

    logger.info('Computing layout (ForceAtlas2)')
    forceatlas2 = ForceAtlas2(
        outboundAttractionDistribution=True,
        edgeWeightInfluence=1.0,
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0,
        verbose=False
    )

    positions = forceatlas2.forceatlas2_networkx_layout(graph)

    logger.info('Rendering graph with matplotlib')
    cmap = cm.get_cmap('winter', max(partition.values()) + 1)

    degrees = dict(graph.degree)
    max_degree = max(degrees.values())
    size_multipliers = {i: degrees[i] / max_degree for i in positions}

    # Generate node sizes
    node_size = [
        max(max_node_size * size_multipliers[i], min_node_size)
        for i in positions
    ]

    # Draw the nodes
    nx.draw_networkx_nodes(
        graph,
        positions,
        partition.keys(),
        node_size=node_size,
        cmap=cmap,
        node_color=list(partition.values()),
        alpha=1
    )

    # Draw the edges with a bezier curve
    curves = curved_edges(graph, positions)
    # Assign a colour to each edge, based on the community of the source node.
    edge_color =  [cmap(partition[a]) for a, _ in graph.edges]
    edge_lines = LineCollection(curves, color=edge_color, cmap=cmap, alpha=0.05, linewidths=1)
    plt.gca().add_collection(edge_lines)

    # Draw node labels (words)
    for i, (x, y) in positions.items():
        # The size of the label is proportional to the degree of the node.
        fontsize = max(max_font_size * size_multipliers[i], min_font_size)
        plt.text(x, y, embeddings.words[i], fontsize=fontsize, ha='center', va='center')


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

    if not args.verbose:
        logger.setLevel(logging.ERROR)

    embeddings = WordEmbeddings(
        weights_filepath, vocab_filepath,
        name_metadata=weights_filepath.parent.stem
    )

    figsize = (args.figure_width / args.figure_dpi, args.figure_height / args.figure_dpi)
    plt.figure(figsize=figsize, dpi=args.figure_dpi)

    draw_k_hop_graph(
        embeddings,
        args.target_word,
        args.k,
        alpha=args.alpha,
        min_node_size=args.min_node_size,
        max_node_size=args.max_node_size,
        min_font_size=args.min_font_size,
        max_font_size=args.max_font_size
    )

    # Show the plot, or output it, depending on the mode.
    plt.axis('off')
    if not args.output_path:
        plt.show()
    else:
        output_format = (args.output_path.suffix or 'png').replace('.', '')
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == 'tex' or output_format == 'latex':
            import tikzplotlib
            tikzplotlib.save(args.output_path)
        else:
            plt.savefig(args.output_path, dpi=args.export_dpi)
        logger.info('Exported figure to {}'.format(args.output_path))


if __name__ == '__main__':
    # import python_ta
    # python_ta.check_all(config={
    #     'extra-imports': [
    #         'tqdm',
    #         'logging',
    #         'argparse',
    #         'pathlib',
    #         'typing',
    #         'numpy',
    #         'networkx',
    #         'matplotlib.cm',
    #         'matplotlib.pyplot',
    #         'matplotlib.collections',
    #         'community',
    #         'fa2',
    #         'sklearn.metrics',
    #         'curved_edges',
    #         'word_embeddings',
    #         'logger',
    #     ],
    #     'allowed-io': [''],
    #     'max-line-length': 100,
    #     'disable': ['R1705', 'C0200', 'W0612']
    # })

    parser = argparse.ArgumentParser(description='A tool for visualising k-hop graphs.')
    parser.add_argument('target_word', type=str, help='The word of interest.')
    # Preview and export configuration
    parser.add_argument('-o', '--output', dest='output_path', type=Path, default=None,
                        help='The file to write the figure to.')
    parser.add_argument('-fw', '--figure-width', type=int, default=800,
                        help='The width of the exported file.')
    parser.add_argument('-fh', '--figure-height', type=int, default=600,
                        help='The heght of the exported file.')
    parser.add_argument('-dpi', '--figure-dpi', type=int, default=96,
                        help='The DPI of the exported file.')
    parser.add_argument('-edpi', '--export-dpi', type=int, default=96,
                        help='The DPI of the exported file.')
    parser.add_argument('--verbose', action='store_true', help='Whether to log messages.')
    # Word Embeddings location
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
    # K-hop Graph configuration
    parser.add_argument('--k', type=int, default=2, help='The number of \'hops\' between '
                        'the word of interest and every node in the graph.')
    parser.add_argument('--alpha', type=float, default=None,
                        help='The similarity threshold. If unspecified, defaults to 0.05 '
                        'less than the cosine similarity of the most similar word to the '
                        'word of interest.')
    parser.add_argument('--min-node-size', type=float, default=20,
                        help='The minimum size of a node.')
    parser.add_argument('--max-node-size', type=float, default=120,
                        help='The maximum size of a node.')
    parser.add_argument('--min-font-size', type=float, default=6,
                        help='The minimum size of a label.')
    parser.add_argument('--max-font-size', type=float, default=24,
                        help='The minimum size of a label.')
    main(parser.parse_args())
