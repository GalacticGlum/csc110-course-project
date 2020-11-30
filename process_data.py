"""
Process raw Twitter data.

This script recursively searches for .bz2 files in the given
directory, decompressses them, and filters the JSON Twitter data.

It is ASSUMED that the data is a JSON file containing a list of
JSON objects describing each tweet. The tweet JSON objects should
be in the form returned by the Twitter API.

See https://developer.twitter.com/en/docs for data format info.

This script strips out irrelevant metadata, and only keeps:
    - "created_at": the time the tweet was created.
    - "id": the id of the tweet as assigned by Twitter.
    - "text": the tweet text.
    - "truncated": boolean indicating whether the text was truncated.
    - "coordinates" (if not null): location of client as a geoJSON object.
    - "place" (if not null): a Place object where the tweet is associated.
    - "quote_count" (if not null or 0): approximately how many times the tweet has been quoted.
    - "reply_count" (if not null or 0): number of times the tweet has been replied to.
    - "retweet_count" (if not null or 0): number of times the tweet has been retweeted.
    - "favorite_count" (if not null or 0): approximately how many times the tweet has been liked.
    - "lang": the BCP 47 language id as corresponding to the machine-detected
              language of the tweet, or "und" if no language could be detected.

Note that all data processing is done in-memory, so as to not
use additional hard disk space for processing the data. If processing
large amounts of data (> 100 GB), at least 8 GB of RAM is recommended.

Running this script with the --delete-source flag is a DESTRUCTIVE and IRREVERSIBLE operation.
DO SO AT YOUR OWN RISK!

We delete the source bz2 file before saving the cleaned tweets to (ideally) free up space for the
output JSON file. But, this means that if there was an error, the source is gone.
"""

import bz2
import json
import argparse

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Set

from texttable import Texttable
from hurry.filesize import size

from utils import parallel_map


def clean_tweet(tweet_data: dict) -> dict:
    """Clean a tweet object by removing irrelevant metadata.

    It is assumed that the tweet_data object is in the format returned by the
    Twitter API and that the id and text attributes are not missing.

    Preconditions:
        - 'id' in tweet_data
        - 'text' in tweet_data
    """
    result = {
        'id': tweet_data['id'],
        'text': tweet_data['text'],
        'truncated': tweet_data.get('truncated', False),
        'lang': tweet_data.get('lang', None),
        'created_at': tweet_data.get('created_at', None)
    }

    # The attributes to copy which require checking if they are NULL or ZERO.
    ATTRIBUTES_TO_COPY = [
        'coordinates', 'place', 'quote_count',
        'reply_count', 'retweet_count', 'favorite_count'
    ]

    for attribute_name in ATTRIBUTES_TO_COPY:
        value = tweet_data.get(attribute_name, None)
        # Include if it is not NULL OR ZERO
        if value is None or value == 0:
            continue

        result[attribute_name] = value

    return result


def is_retweet(tweet_data: dict) -> bool:
    """Return whether the tweet object represents a retweet."""
    return tweet_data.get('retweeted_status', None) is not None


def is_valid_tweet(tweet_data: dict, language_filters: Set[str]) -> bool:
    """Return whether the tweet is valid.

    Preconditions:
        - language_filters is not None
    """
    # Make sure the tweet has required attributes and isn't deleted
    if 'delete' in tweet_data or \
       'id' not in tweet_data or \
       'text' not in tweet_data:
        return False

    return len(language_filters) == 0 or tweet_data.get('lang', None) in language_filters


@dataclass
class ProcessedStatistics:
    """Statistics about a processed file.

    Instance Attributes:
        - source_size: the size of the source file in bytes.
        - processed_size: the size of the processed file in bytes.
        - num_tweets_source: the number of tweets in the source file.
        - num_tweets_processed: the number of tweets after cleaning/filtering.
    """
    source_size: int
    processed_size: int
    num_tweets_source: int
    num_tweets_processed: int


def load_tweets(filepath: Path) -> Tuple[List[dict], int]:
    """Load tweets from a bz2 file.
    Return a list of tweet data and the size of the source file, in bytes."""
    with open(filepath, 'rb') as file:
        # Decompress file and load it into memory
        data = bz2.decompress(file.read())
        source_size_decompressed = len(data)

        # Convert the bytearray to a string
        data = data.decode('utf-8').strip()

    # At this point, this should just be a string containing JSON data.
    # So, we can change load it into a Python dict.
    #
    # One quirk with the Twitter archive dataset is that some files contain
    # a JSON list, while others contain each object on its own line.
    # This second format is not technically 'legal' JSON, since you can't
    # have multiple root objects. But, we will handle both.
    if data[0] == '[' and data[-1] == ']':
        # If the file starts and ends with list 'characters' then we can be
        # reasonably sure that the JSON data is formatted as a list.
        tweets = json.loads(data)
    else:
        # Split by line and load each object separately.
        tweets = [json.loads(line) for line in data.split('\n')]

    return tweets, source_size_decompressed


def process_file(filepath: Path, language_filters: Set[str] = None,
                 keep_retweets: bool = False, delete_source: bool = False,
                 destination_directory: Path = None) -> ProcessedStatistics:
    """Process a .bz2 file based on the given language filters."""
    # Load data
    tweets, source_size_decompressed = load_tweets(filepath)

    # Filter data
    language_filters = language_filters or set()
    cleaned_tweets = []
    for tweet in tweets:
        if not is_valid_tweet(tweet, language_filters):
            continue

        retweet = is_retweet(tweet)
        if retweet:
            # For retweets, we get the original tweet.
            orignal_tweet = clean_tweet(tweet['retweeted_status'])
            cleaned_tweets.append(orignal_tweet)

        if not retweet or retweet and keep_retweets:
            cleaned_tweets.append(clean_tweet(tweet))

    if delete_source:
        try:
            # This is a risky and irreversible operation!
            # We delete the source before saving the cleaned tweets
            # to ideally free up space for the output file. But,
            # this means that if there was an error, the source is gone.
            filepath.unlink()
        except IOError:
            # It's not a big deal if we couldn't delete the source.
            # We should just make a note of it (via logging) and move on.
            pass

    destination_filepath = filepath.with_suffix('')
    if destination_filepath.suffix != '.json':
        destination_filepath.with_suffix('.json')

    if destination_directory is not None:
        destination_directory.mkdir(exist_ok=True, parents=True)
        destination_filepath = destination_directory / destination_filepath.name

    with open(destination_filepath, 'w+') as output_file:
        json.dump(cleaned_tweets, output_file)

    return ProcessedStatistics(
        source_size_decompressed,  # Size of source in bytes
        destination_filepath.stat().st_size,  # Size of processed in bytes
        len(tweets),  # Number of tweets in the source file
        len(cleaned_tweets)  # Number of tweets after filtering/cleaning
    )


def main(args: argparse.Namespace) -> None:
    """Main entrypoint for the script."""
    # Get all files as a list
    files = list(args.input_directory.glob(args.glob))
    # Convert language filters to set for fast lookup
    language_filters = set(args.language_filters)

    process_file_kwargs = {
        'language_filters': language_filters,
        'keep_retweets': args.keep_retweets,
        'delete_source': args.delete_source,
        'destination_directory': args.destination_directory
    }

    # A list containing the output of the process_file
    # function for each element of files.
    sizes = parallel_map(
        [{'filepath': file, **process_file_kwargs} for file in files],
        process_file,
        n_jobs=args.num_workers,
        use_kwargs=True,
        include_errors=False
    )

    if args.summarise:
        # The parallel_map returns the output of the function, or an Exception
        # (if one was raised). So, we need to filter out all invalid values.
        sizes = [x for x in sizes if isinstance(x, ProcessedStatistics)]

        # Compute total stats
        total_source_size = sum(stats.source_size for stats in sizes)
        total_processed_size = sum(stats.processed_size for stats in sizes)
        total_source_tweets = sum(stats.num_tweets_source for stats in sizes)
        total_processed_tweets = sum(stats.num_tweets_processed for stats in sizes)

        print(f'Processed a total of {len(sizes)} files.\n')
        table = Texttable()
        table.set_deco(Texttable.HEADER)
        table.add_rows(
            [['', 'raw', 'cleaned'],
             ['total size', size(total_source_size), size(total_processed_size)],
             ['total tweets', total_source_tweets, total_processed_tweets]]
        )
        print(table.draw())


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'extra-imports': [
            'bz2',
            'json',
            'argparse',
            'pathlib',
            'dataclasses',
            'texttable',
            'hurry.filesize',
            'utils'
        ],
        'allowed-io': ['main', 'process_file', 'load_tweets'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    parser = argparse.ArgumentParser(description='Process raw Twitter data.')
    parser.add_argument('input_directory', type=Path, help='Directory containing the data.')
    parser.add_argument('-o', '--destination-directory', type=Path, default=None,
                        help='The directory to save the files. If unspecified, '
                        'defaults to the location of the processed bz2 file.')
    parser.add_argument('-d', '--delete-source', action='store_true', dest='delete_source',
                        help='Whether to delete the source .bz2 files after processing.')
    parser.add_argument('-l', '--language-filters', nargs='+', type=str, default=['en'],
                        help='Languages to include. If empty, no language filtering is applied.')
    parser.add_argument('--keep-retweets', action='store_true', dest='keep_retweets',
                        help='Keep tweets that have a "retweeted_status" param.')
    parser.add_argument('--glob', type=str, default='**/*.bz2',
                        help='Glob pattern to find data files.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='The number of threads to use. Defaults to 8.')
    parser.add_argument('--summarise', action='store_true', dest='summarise',
                        help='Display a summary of the data processing.')
    parser.set_defaults(keep_retweets=False)
    main(parser.parse_args())
