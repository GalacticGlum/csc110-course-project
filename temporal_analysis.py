"""Functionality for performing temporal analysis on a corpus."""

from pathlib import Path
from datetime import date
from typing import Union, Tuple, List
import matplotlib.pyplot as plt

from word2vec import Tokenizer
from word_embeddings import WordEmbeddings, cosine_similarity

def plot_similarity_over_time(temporal_embeddings: List[Tuple[date, WordEmbeddings]],
                              word_a: str, word_b: str) -> None:
    """Plot the cosine similarity of a word pair over time.
    This function does NOT call show on the resultant plot.

    Args:
        temporal_embeddings: A list of 2-tuples containing a date and word embeddings.
        word_a: The first word to compare.
        word_b: The second word to compare.
    """
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


def plot_frequency_over_time(temporal_tokenizers: List[Tuple[date, Tokenizer]],
                             words: Union[str, List[str]]) -> None:
    """Plot the frequency of the given words over time.
    This function does NOT call show on the resultant plot.

    Args:
        temporal_tokenizers: A list of 2-tuples containing a date and tokenizer.
        words: One or more words in the vocabulary whose frequency to plot.
    """
    # Make words a list (for consistency)
    if not isinstance(words, list):
        words = [words]

    for word in words:
        x, y = [], []
        for date, tokenizer in temporal_tokenizers:
            if word not in tokenizer.vocabulary:
                continue
            x.append(date)
            y.append(tokenizer.get_frequency(word))
        plt.plot_date(x, y, '-o')

    def readable_list(seq: List[any]) -> str:
        seq = [str(s) for s in seq]
        if len(seq) <= 2:
            return ' and '.join(seq)
        return ', '.join(seq[:-1]) + ', and ' + seq[-1]

    word_list = readable_list([f'"{x}"' for x in words])
    plt.title(f'Mentions of the terms {word_list}')
    plt.xlabel('Date')
    plt.ylabel('Number of mentions')


if __name__ == '__main__':
    from dateutil.parser import parse
    paths = Path('./output/word2vec').glob('00001-20*')
    # temporal_embeddings = []
    temporal_tokenizers = []
    for path in paths:
        if path.stem == '00001-2018_09_p25_corpus': continue

        path_name = path.stem
        path_name = path_name[path_name.find('-') + 1:path_name.find('_p')].replace('_', '-')

        path_date = parse(path_name, fuzzy=True).replace(day=1)

        # Load the embeddings
        # embeddings = WordEmbeddings(
        #     path / 'proj_weights.npy',
        #     path / 'vocab.txt',
        #     suffix_tree=False,
        #     nearest_neighbours=False
        # )

        # Load the tokenizer
        tokenizer = Tokenizer()
        tokenizer.load(path / 'tokenizer.json')

        # temporal_embeddings.append((path_date.date(), embeddings))
        temporal_tokenizers.append((path_date.date(), tokenizer))

    # plot_similarity_over_time(temporal_embeddings, 'climate', 'homelessness')
    plot_frequency_over_time(temporal_tokenizers, 'climate')
    plt.show()