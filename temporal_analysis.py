"""Functionality for performing temporal analysis on a corpus."""

from pathlib import Path
from datetime import date
from typing import Tuple, List
import matplotlib.pyplot as plt
from word_embeddings import WordEmbeddings, cosine_similarity

def plot_similarity_over_time(temporal_embeddings: List[Tuple[date, WordEmbeddings]],
                              word_a: str, word_b: str) -> None:
    """Plot the cosine similarity of a word pair over time.
    This function does NOT call show on the resultant plot.

    Args:
        temporal_embeddings: A list of 2-tuples containing a date and word embeddings.
            The datetime represents the date associated with the word embeddings.
        word_a: The first word to compare.
        word_b: The second word to compare.
    """
    # Sort the embeddings in increasing order of date
    temporal_embeddings.sort(key=lambda x: x[0])
    x, y = [], []
    for date, embeddings in temporal_embeddings:
        if word_a not in embeddings._vocabulary or \
           word_b not in embeddings._vocabulary:
           continue

        x.append(date)
        # Get embedding vectors for each word
        u = embeddings.get_vector(word_a)
        v = embeddings.get_vector(word_b)
        similarity = cosine_similarity(u, v)
        y.append(similarity)

    plt.title(f'{word_a} - {word_b}')
    plt.xlabel('Date')
    plt.ylabel('Cosine Similarity')
    plt.plot_date(x, y, '-o')


if __name__ == '__main__':
    from dateutil.parser import parse
    paths = Path('./output/word2vec').glob('00001-20*')
    temporal_embeddings = []
    for path in paths:
        path_name = path.stem
        path_name = path_name[path_name.find('-') + 1:path_name.find('_p')].replace('_', '-')

        path_date = parse(path_name, fuzzy=True).replace(day=1)

        # Load the embeddings
        embeddings = WordEmbeddings(
            path / 'proj_weights.npy',
            path / 'vocab.txt',
            suffix_tree=False,
            nearest_neighbours=False
        )

        temporal_embeddings.append((path_date.date(), embeddings))

    plot_similarity_over_time(temporal_embeddings, 'climate', 'homelessness')
    plt.show()