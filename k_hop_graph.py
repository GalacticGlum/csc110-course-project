"""A tool for visualising k-hop graphs."""

from typing import Set, Optional

import tqdm
import numpy as np
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import community as community_louvain
from matplotlib.collections import LineCollection

from sklearn import metrics
from fa2 import ForceAtlas2
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
        graph.add_edge(i, j)

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


# G = nx.random_geometric_graph(400, 0.2)
embeddings = WordEmbeddings(
    'output/word2vec/00001-shakespeare/proj_weights.npy',
    'output/word2vec/00001-shakespeare/vocab.txt'
)
G = build_k_hop_graph(embeddings, 'juliet', 2, alpha=0.89)
# compute the best partition
print('computing best partition')
partition = community_louvain.best_partition(G)

forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=True,  # Dissuade hubs
                        edgeWeightInfluence=1.0,
                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=1.2,
                        # Tuning
                        scalingRatio=2.0,
                        strongGravityMode=False,
                        gravity=1.0,
                        verbose=True)

print('computing forceatlas2 layout')
positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)

print('drawing')
cmap = cm.get_cmap('tab20b', max(partition.values()) + 1)
edge_color =  [cmap(partition[a]) for a, b in G.edges]

curves = curved_edges(G, positions)
lc = LineCollection(curves, color=edge_color, cmap=cmap, alpha=0.25, linewidths=1)

nx.draw_networkx_nodes(G, positions, partition.keys(), node_size=20, cmap=cmap, node_color=list(partition.values()), alpha=1)
plt.gca().add_collection(lc)
# nx.draw_networkx_edges(G, positions, edge_cmap=cmap, edge_color=edge_color, alpha=0.05)
plt.axis('off')
plt.show()