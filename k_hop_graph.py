"""A tool for visualising k-hop graphs."""

from typing import Set, Optional

import tqdm
import numpy as np
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import community as community_louvain
from matplotlib.collections import LineCollection

from fa2 import ForceAtlas2
from sklearn import metrics
from curved_edges import curved_edges
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
    print('generating infinity-hop graph')
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
    print('computing shortest paths')
    paths = nx.single_source_shortest_path_length(graph, T)

    print('building k-hop graph')
    nodes_to_delete = set()
    for node in tqdm.tqdm(graph.nodes):
        # Remove the node if the word of interest is not reachable in at most k edges.
        if node not in paths or paths[node] > k:
            nodes_to_delete.add(node)

    for node in nodes_to_delete:
        graph.remove_node(node)

    print(f'Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}')
    return graph


def draw_k_hop_graph(embeddings: WordEmbeddings, target_word: str,
                     k: int, alpha: Optional[float] = 0.50,
                     min_node_size: Optional[float] = 20,
                     max_node_size: Optional[float] = 120,
                     min_font_size: Optional[float] = 6,
                     max_font_size: Optional[float] = 24) -> None:
    """Draw the k-hop graph for the given word embeddings and interest word.

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
    graph = build_k_hop_graph(embeddings, target_word, k, alpha=alpha)

    print('computing best partition')
    # compute the best partition
    partition = community_louvain.best_partition(graph)

    print('computing layout')
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

    print('drawing')

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

    plt.axis('off')
    plt.show()


embeddings = WordEmbeddings(
    'output/word2vec/00001-shakespeare/proj_weights.npy',
    'output/word2vec/00001-shakespeare/vocab.txt'
)

draw_k_hop_graph(embeddings, 'juliet', 3, alpha=0.88)