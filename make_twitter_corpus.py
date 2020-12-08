"""Make a corpus from Twitter JSON data.

This script recursively searches for .json files in the given
directory, and adds the text of each tweet object in the JSON
file to a new line in the specified output text file.

It is ASSUMED that the data in the JSON file is a list consisting
of JSON objects, where each object has a "text" attribute. In general,
the tweet JSON objects should be in the form returned by the Twitter API
or from the clean_twitter_data.py script (refer to the docstring for more info).
"""

import re
import json
import random
import argparse
from pathlib import Path
from typing import Optional, List, IO

from tqdm import tqdm
from logger import logger
from utils import parallel_map

# A Regex pattern to match urls starting with or without http(s).
URL_MATCH_PATTERN = re.compile(
    r'(?i)(https?:\/\/(?:www\.|(?!www))'
    r'[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'
    r'www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'
    r'https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|'
    r'www\.[a-zA-Z0-9]+\.[^\s]{2,})'
)


def get_tweet_texts(filepath: Path,
                    remove_newlines: Optional[bool] = True,
                    remove_unicode: Optional[bool] = True,
                    strip_tweet_text: Optional[bool] = True,
                    ignore_truncated_tweets: Optional[bool] = False,
                    remove_links: Optional[bool] = True,
                    special_phrase_pattern: Optional[str] = None) \
        -> List[str]:
    """Return a list of tweet texts from a JSON file.

    Args:
        filepath: The path to the JSON file containing the tweet data.
        remove_newlines: Whether to replace newline characters with spaces.
        remove_unicode: Whether to remove unicode characters.
        strip_tweet_text: Whether to strip the text (remove leading and
            trailing whitespace from the tweet texts).
        ignore_truncated_tweets: Whether to ignore truncated tweets.
        remove_links: Whether to remove links.
        special_phrase_pattern: Regex pattern to match special phrases.
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

            if remove_links:
                text = re.sub(URL_MATCH_PATTERN, '', text)

            if special_phrase_pattern is not None:
                for match in re.finditer(special_phrase_pattern, text):
                    i, j = match.span()
                    # Replace spaces in the special phrases with underscores.
                    text = text[:i] + text[i:j].replace(' ', '_') + text[j:]

            if strip_tweet_text:
                text = text.strip()

            texts.append(text)
    return texts


def process_file(data_filepath: Path, fp: IO,
                 proportion: Optional[float] = 1,
                 remove_links: Optional[bool] = True,
                 special_phrase_pattern: Optional[str] = None) -> None:
    """Process a file.

    Args:
        data_filepath: The path to the JSON file containing the tweet data.
        fp: A file-like objet to write to.
        proportion: The proportion of the tweets to save.
            Randomly samples the given proportion of the loaded tweets.
        remove_links: Whether to remove links.
        special_phrase_pattern: Regex pattern to match special phrases.

    Preconditions:
        - 0 < proportion <= 1
    """
    texts = get_tweet_texts(
        data_filepath,
        remove_links=remove_links,
        special_phrase_pattern=special_phrase_pattern
    )

    texts = random.sample(texts, k=int(len(texts) * proportion))
    fp.write('\n'.join(texts))


def main(args: argparse.Namespace) -> None:
    """Main entrypoint for the script."""
    # Initialize seed.
    random.seed(args.seed)

    special_phrases = []
    if args.special_phrases_path is not None:
        with open(args.special_phrases_path) as special_phrases_file:
            special_phrases = json.load(special_phrases_file)

    # A pattern to match any of the special phrases
    special_phrase_pattern = re.compile('|'.join(
        '({})'.format(re.escape(phrase))
        for phrase in special_phrases
    ), flags=re.IGNORECASE)

    # Get all files as a list
    files = list(args.input_directory.glob('**/*.json'))
    with open(args.output_file, 'w+', encoding='utf-8') as output_file:
        proportion = max(0, min(1, args.proportion))
        process_file_kwargs = {
            'fp': output_file,
            'proportion': proportion,
            'remove_links': args.remove_links,
            'special_phrase_pattern': special_phrase_pattern
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
    parser.add_argument('-r', '--no-remove-links', dest='remove_links', action='store_false',
                        help='Don\'t remove links from tweets.')
    parser.add_argument('--special-phrases', dest='special_phrases_path', type=Path, default=None,
                        help='A JSON file containing a list of special phrases to encode. '
                             'Special phrases are encoded by inserting underscores for spaces.')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Sets the seed of the random engine.')
    main(parser.parse_args())