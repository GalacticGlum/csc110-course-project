"""Make a corpus from Twitter JSON data.

This script recursively searches for .json files in the given
directory, and adds the text of each tweet object in the JSON
file to a new line in the specified output text file.

It is ASSUMED that the data in the JSON file is a list consisting
of JSON objects, where each object has a "text" attribute. In general,
the tweet JSON objects should be in the form returned by the Twitter API
or from the clean_twitter_data.py script (refer to the docstring for more info).
"""

import json
import random
import argparse
from pathlib import Path
from typing import Optional, List, IO

from tqdm import tqdm
from logger import logger
from utils import parallel_map


def get_tweet_texts(filepath: Path,
                    remove_newlines: Optional[bool] = True,
                    remove_unicode: Optional[bool] = True,
                    strip_tweet_text: Optional[bool] = True,
                    ignore_truncated_tweets: Optional[bool] = False) \
        -> List[str]:
    """Return a list of tweet texts from a JSON file.

    Args:
        filepath: The path to the JSON file containing the tweet data.
        remove_newlines: Whether to replace newline characters with spaces.
        remove_unicode: Whether to remove unicode characters.
        strip_tweet_text: Whether to strip the text (remove leading and
            trailing whitespace from the tweet texts).
        ignore_truncated_tweets: Whether to ignore truncated tweets.
    """
    texts = []
    with open(filepath) as file:
        tweets = json.load(file)
        for tweet in tweets:
            # Make sure the object has a "text" attribute
            if 'text' not in tweet:
                continue

            # Ignore the tweet if it is truncated and the flag is set.
            if ignore_truncated_tweets and tweet.get('truncated', False):
                continue

            text = tweet['text']
            if remove_newlines:
                # Replace newlines with spaces
                text = text.replace('\n', ' ').replace('\r', '')

            if remove_unicode:
                text = str(text.encode('utf-8').decode('ascii', 'ignore'))

            if strip_tweet_text:
                text = text.strip()

            texts.append(text)
    return texts


def process_file(data_filepath: Path, fp: IO,
                 proportion: Optional[float] = 1) -> None:
    """Process a file.

    Args:
        data_filepath: The path to the JSON file containing the tweet data.
        fp: A file-like objet to write to.
        proportion: The proportion of the tweets to save.
            Randomly samples the given proportion of the loaded tweets.

    Preconditions:
        - 0 < proportion <= 1
    """
    texts = get_tweet_texts(data_filepath)
    texts = random.sample(texts, k=int(len(texts) * proportion))
    fp.write('\n'.join(texts))


def main(args: argparse.Namespace) -> None:
    """Main entrypoint for the script."""
    # Initialize seed.
    random.seed(args.seed)

    # Get all files as a list
    files = list(args.input_directory.glob('**/*.json'))
    with open(args.output_file, 'w+', encoding='utf-8') as output_file:
        proportion = max(0, min(1, args.proportion))
        process_file_kwargs = {
            'fp': output_file,
            'proportion': proportion
        }

        parallel_map(
            [{'data_filepath': file, **process_file_kwargs} for file in files],
            process_file,
            n_jobs=args.num_workers,
            use_kwargs=True,
            return_output=False
        )


if __name__ == '__main__':
    # import python_ta
    # python_ta.check_all(config={
    #     'extra-imports': [
    #         'json',
    #         'random',
    #         'argparse',
    #         'pathlib',
    #         'tqdm',
    #         'logger',
    #         'utils'
    #     ],
    #     'allowed-io': ['main'],
    #     'max-line-length': 100,
    #     'disable': ['R1705', 'C0200']
    # })

    parser = argparse.ArgumentParser(description='Make a corpus from Twitter JSON data.')
    parser.add_argument('input_directory', type=Path, help='Directory containing the data.')
    parser.add_argument('output_file', type=Path, help='The output filename.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='The number of threads to use. Defaults to 8.')
    parser.add_argument('-p', '--proportion', type=float, default=1,
                        help='The proportion of the tweets to save.')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Sets the seed of the random engine.')
    main(parser.parse_args())