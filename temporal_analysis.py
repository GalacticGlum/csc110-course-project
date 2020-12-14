"""Functionality for performing temporal analysis on a corpus."""

from pathlib import Path
from datetime import date, datetime
from typing import Optional, Union, Tuple, List

import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_text as text
import matplotlib.pyplot as plt

from utils import list_join
from word2vec import Tokenizer
from train_text_classifier import build_classifier_model
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
                             words: Union[str, List[str]], proportion: Optional[bool] = False) \
        -> None:
    """Plot the frequency of the given words over time.
    This function does NOT call show on the resultant plot.

    Args:
        temporal_tokenizers: A list of 2-tuples containing a date and tokenizer.
        words: One or more words in the vocabulary whose frequency to plot.
        proportion: Whether to plot the raw frequency, or the proportion.
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
            yi = tokenizer.get_frequency(word)
            if proportion:
                yi /= tokenizer.corpus_size

            y.append(yi)
        plt.plot_date(x, y, '-o')

    word_list = list_join([f'"{x}"' for x in words])
    variable_label = 'Proportion' if proportion else 'Mentions'
    plt.title(f'{variable_label} of the terms {word_list}')
    plt.xlabel('Date')

    y_label = 'Proportion' if proportion else 'Number of mentions'
    plt.ylabel(y_label)


def get_sentiment_over_time(temporal_tweets: List[Tuple[datetime, List[str]]],
                            model: Optional[object] = None, restore_dir: Optional[Path] = None,
                            classes: List[int] = [-1, 0, 2, 1],
                            stat_func: Optional[callable] = None,
                            show_progress_bar: Optional[bool] = True) -> List[Tuple[date, float]]:
    """Compute the climate change sentiment of the given tweets over time.
    Requires one of model or restore_dir.

    Args:
        model: A loaded climate change sentiment classification model.
        restore_dir: The checkpoint to load for the climate change sentiment classification model.
        temporal_tweets: A list of 2-tuples of tweet collections, where the first element is the
            datetime of the tweets, and the second element is a list of tweet texts.

            For every 'bucket' of tweets, a statitic about sentiment score is calculated,
            and then plotted. For example, the mean or max sentiment score in the list of tweeets.
        stat_func: A function which takes in an array-like of sentiment scores, and their
            probability as given by the text classification model, and then outputs a scalar
            value describing the sentiment scores (i.e. a summary statistic).

            Defaults to the mean sentiment score as a weighted sum of their probabilities:
            s = dot(S, P) / n, where S is a vector of the sentiment scores, P is a vector
            of their probabilities, and n = |S| = |P| is the dimensionality of the vectors.
        classes: A mapping from class label (output of the model) to a sentiment class (e.g. 1, 0, -1
            for positive, netural, and negative respectively). Defaults to [-1, 0, 2, 1] which are the
            sentiment values for the climate change sentiment dataset.
        show_progress_bar: Whether to show a progress bar while the jobs run.
    """
    def _weighted_mean_score(sentiments: tf.Tensor, probabilities: tf.Tensor):
        """Compute the weighted mean sentiment score.

        Args:
            sentiments: An int tensor of sentiment classes predicted by the model.
            probabilities: A float tensor of the probability of each class (i.e. model confidence).

        Preconditions:
            - sentiments.shape == probabilities.shape
        """
        return np.dot(sentiments, probabilities) / sentiments.shape[-1]

    # Set stat_func to default if not specified
    stat_func = _weighted_mean_score or stat_func

    # Load sentiment classification model (if one not provided)
    if model is None and restore_dir is not None:
        model = tf.saved_model.load(restore_dir)
    elif model is None and restore_dir is None:
        raise ValueError('Requires one of model or restore_dir.')

    results = []
    class_tensor = tf.convert_to_tensor(classes, dtype=tf.float32)
    for timestamp, tweets in tqdm.tqdm(temporal_tweets, disable=not show_progress_bar):
        outputs = tf.sigmoid(model(tf.constant(tweets)))
        # class_matrix = tf.reshape(tf.tile(class_tensor, [outputs.shape[0]]), (outputs.shape[0], class_tensor.shape[0]))
        # yi = tf.reduce_mean(tf.reduce_sum(tf.multiply(outputs, class_matrix), 1)).numpy()
        sentiments = tf.gather(classes, tf.argmax(outputs, axis=1))
        probabilities = tf.reduce_max(outputs, axis=1)

        yi = stat_func(sentiments, probabilities)
        print(timestamp, yi)
        results.append((timestamp, yi))
    return results

# if __name__ == '__main__':
#     from dateutil.parser import parse
#     paths = Path('./output/word2vec').glob('00001-20*')
#     # temporal_embeddings = []
#     temporal_tokenizers = []
#     for path in paths:
#         path_name = path.stem
#         path_name = path_name[path_name.find('-') + 1:path_name.find('_p')].replace('_', '-')

#         path_date = parse(path_name, fuzzy=True).replace(day=1)

#         # Load the embeddings
#         # embeddings = WordEmbeddings(
#         #     path / 'proj_weights.npy',
#         #     path / 'vocab.txt',
#         #     suffix_tree=False,
#         #     nearest_neighbours=False
#         # )

#         # Load the tokenizer
#         tokenizer = Tokenizer()
#         tokenizer.load(path / 'tokenizer.json')

#         # temporal_embeddings.append((path_date.date(), embeddings))
#         temporal_tokenizers.append((path_date.date(), tokenizer))

#     # plot_similarity_over_time(temporal_embeddings, 'climate', 'homelessness')
#     plot_frequency_over_time(temporal_tokenizers, ['climate_change', 'global_warming', 'disaster'], proportion=True)
#     plt.show()


def random_sampler(filename, k):
    sample = []
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()

        random_set = sorted(random.sample(range(filesize), k))

        for i in range(k):
            f.seek(random_set[i])
            # Skip current line (because we might be in the middle of a line)
            f.readline()
            # Append the next line to the sample set
            sample.append(f.readline().rstrip())

    return sample


if __name__ == '__main__':
    from dateutil.parser import parse
    import random
    files = Path('./data/twitter/').glob('20*.txt')
    temporal_tweets = []
    for file in files:
        print(file)
        filename = file.stem
        filename = filename[filename.find('-') + 1:filename.find('_p')].replace('_', '-')
        file_date = parse(filename, fuzzy=True).replace(day=1)

        # Get tweets (1K per month to run sentiment analysis)
        tweets = random_sampler(file, k=1000)
        temporal_tweets.append((file_date, tweets))

    model = tf.saved_model.load('output/classifier/00003-climate_change_sentiment/model_final')
    x = get_sentiment_over_time(temporal_tweets, model=model)
    print(x)