"""The main entrypoint for the project.
This module provides a few different ways of running the code.
"""
from pathlib import Path
from typing import Union
from word_embeddings import WordEmbeddings
from k_hop_graph import visualise_k_hop_graph
from embedding_projector import embedding_projector

def run_embedding_projector(root_directory: Union[str, Path]) -> None:
    """Run the embedding projector on ALL the trained embeddings in the output/word2vec folder.

    Args:
        root_directory: The directory containing all the embedding checkpoints.
    """
    root_directory = Path(root_directory)
    embeddings_list = [
        WordEmbeddings(checkpoint=path)
        for path in root_directory.glob('*/')
    ]
    embedding_projector(embeddings_list)


if __name__ == '__main__':
    # For this to work, you need AT LEAST ONE trained embeddings file
    # (i.e. a proj_weights.npy and vocab.txt pair).
    #
    # For a minimal working example, download a subset of the trained data from
    # the link provided in the paper.

    # Run the embedding projector on all the embeddings in the output/word2vec folder.
    # run_embedding_projector('./output/word2vec/')

    # We can visualise the k-hop graph for a word of interest and embedding space.
    #
    # Simply specify the directory of the checkpoint to use (for example,
    # output/word2vec/00001-shakespeare), and the word of interest.
    # By default, draw_k_hop_graph will use k = 2, so all nodes in the outputted graph
    # are at most 2 edges from the word of interest. You can change this of course!
    #
    # Depending on the size of the graph produced, you may wish to output it to a file,
    # rather than opening it in a preview window. To do so, simplify specify the output_path
    # (which can be either a string or path.Pathlib file!) argument. Various output formats are
    # supported (png, jpeg, svg, tex to name a few), which are determined by the extension of
    # the output_path.
    #
    # visualise_k_hop_graph(
    #     'romeo',
    #     checkpoint='output/word2vec/00001-shakespeare',
    #     # The similarity score threshold. A lower value will produce more complex/dense graphs.
    #     # WARNING: If alpha is >= than the minimum pairwise cosine similarity between the word of
    #     # interest and any other word in the vocab, then the graph will be empty! The word
    #     # of interest will not be a node of the graph, so there there is no path to it.
    #     alpha=0.89
    # )
