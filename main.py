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
        WordEmbeddings(
            checkpoint_filepath=path,
            name_metadata=path.stem)
        for path in root_directory.glob('*/')
    ]
    # We can't have no embeddings!
    assert len(embeddings_list) > 0
    embedding_projector(embeddings_list)


if __name__ == '__main__':
    # For this to work, you need AT LEAST ONE trained embeddings file
    # (i.e. a proj_weights.npy and vocab.txt pair).
    #
    # For a minimal working example, download a subset of the trained data from
    # the link provided in the paper.

    # Run the embedding projector on all the embeddings in the output/word2vec folder.
    # NOTE: Loading the embeddings may take a bit. Once they have been loaded in,
    # and the server is running, navigate to http://localhost:5006/
    # run_embedding_projector('./output/word2vec/')

    # We can visualise the k-hop graph for a word of interest and embedding space.
    #
    # Simply specify the directory of the checkpoint to use (for example,
    # output/word2vec/00001-shakespeare), and the word of interest.
    # By default, draw_k_hop_graph will use k = 2, so all nodes in the outputted graph
    # are at most 2 edges from the word of interest. You can change this of course!
    #
    # Depending on the size of the graph produced, you may wish to output it to a file,
    # rather than opening it in a preview window. To do so, simply specify the output_path
    # param (which can be either a string or path.Pathlib object!). Various output formats are
    # supported (png, jpeg, svg, tex to name a few), which are determined by the extension of
    # the output_path.

    # visualise_k_hop_graph(
    #     'romeo',
    #     checkpoint='output/word2vec/00001-shakespeare',
    #     # The similarity score threshold. A lower value will produce more complex/dense graphs.
    #     # NOTE: alpha is a very fickle parameter. Small changes in alpha (+/- 0.01) can cause
    #     # reasonably large changes in the resultant k-hop graph. Experimentation is key!
    #     #
    #     # Alternatively, you may omit the alpha parameter entirely, either by commenting
    #     # it out or setting it to None. In this case, a reasonable default will be computed
    #     # based on the given target word and embeddings. For new words of interest, we recommend
    #     # starting with the default and then tuning from there.
    #     #
    #     # WARNING: If alpha is > than the maximum pairwise cosine similarity between the word of
    #     # interest and any other word in the vocab, then the graph will be empty! The word
    #     # of interest will not be a node of the graph, so there there is no path to it.
    #     alpha=0.89,
    #     # output_path='./output/k_hop/romeo_2_hop.png'  # Uncomment me to save to file!,
    #     # Setting verbose to True enables logging, which is nice for progress updates!
    #     verbose=True
    # )
