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
"""


import bz2
import json
import argparse
from tqdm import tqdm
from typing import Set
from pathlib import Path
from texttable import Texttable
from hurry.filesize import size
from dataclasses import dataclass
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed
)


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
                    help='Whether to keep retweet (tweets that have a "retweeted_status" param).')
parser.add_argument('--glob', type=str, default='**/*.bz2', help='Glob pattern to find data files.')
parser.add_argument('--num-workers', type=int, default=8, help='The number of threads to use. Defaults to 8.')
parser.add_argument('--summarise', action='store_true', dest='summarise',
                    help='Display a summary of the data processing.')
parser.set_defaults(keep_retweets=False)


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
        'lang': tweet_data.get('lang', None)
    }

    # The attributes to copy which require checking if they are NULL or ZERO.
    ATTRIBUTES_TO_COPY = [
        'coordinates', 'place', 'quote_count',
        'reply_count', 'retweet_count', 'favorite_count'
    ]

    for attribute_name in ATTRIBUTES_TO_COPY:
        value = tweet_data.get(attribute_name, None)
        # Include if it is not NULL OR ZERO
        if value is None or value == 0: continue
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

    return len(language_filters) == 0 or \
           tweet_data.get('lang', None) in language_filters


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


def process_file(filepath: Path, language_filters: Set[str]=None, \
                 keep_retweets: bool=False, delete_source: bool=False, \
                 destination_directory: Path=None) -> ProcessedStatistics:
    """Process a .bz2 file based on the given language filters.

    Return a ProcessedStatistics object.
    """
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
    try:
        if data[0] == '[' and data[-1] == ']':
            # If the file starts and ends with list 'characters' then we can be
            # reasonably sure that the JSON data is formatted as a list.
            tweets = json.loads(data)
        else:
            # Split by line and load each object separately.
            tweets = [json.loads(line) for line in data.split('\n')]
    except Exception as exception:
        # TODO: log exception
        return

    # Default to empty set if the passed in value was None
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
        except:
            # TODO: log this!
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
        source_size_decompressed, # Size of source in bytes
        destination_filepath.stat().st_size, # Size of processed in bytes
        len(tweets), # number of tweets in the source file
        len(cleaned_tweets) # number of tweets after filtering/cleaning
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

    # sizes is a list of ProcessedStatistics objects.
    sizes = parallel_map(
        [{'filepath': file, **process_file_kwargs} for file in files],
        process_file,
        n_jobs=args.num_workers,
        use_kwargs=True
    )

    if args.summarise:
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
             ['total tweets', total_source_size, total_processed_tweets]]
        )
        print(table.draw())


def parallel_map(array: list, function: callable, n_jobs: int=16, use_kwargs: bool=False,
                 front_num: int=3, multithread: bool=False, show_progress_bar: bool=True,
                 extend_result: bool=False, initial_value: list=list()):
    """
    A parallel version of the map function with a progress bar.
    :note:
        This is a utility function for running parallel jobs with progress
        bar. Originally from http://danshiebler.com/2016-09-14-parallel-progress-bar/.
        The implementation is identical to the source; however, the documentation and
        code style has been modified to fit the style of this codebase.
    :param array:
        An array to iterate over.
    :param function:
        A python function to apply to the elements of array
    :param n_jobs:
        The number of cores to use. Defaults to 16.
    :param use_kwargs:
        Whether to consider the elements of array as dictionaries of
        keyword arguments to function. Defaults to ``False``.
    :param front_num:
        The number of iterations to run serially before kicking off the
        parallel job. Useful for catching bugs
    :param multithread:
        If ``True``, a :class:``concurrent.futures.ThreadPoolExecutor`` will be used rather
        than a :class:``concurrent.futures.ProcessPoolExecutor``. Defaults to ``False``.
    :param show_progress_bar:
        Indicates whether a loading progress bar should be displayed while the process runs.
        Defaults to ``True``.
    :param extend_result:
        Indicates whether the resultant list should be extended rather than appended to.
        Defaults to ``False``. Note that this requires that the return value of ``function``
        is an array-like object.
    :param initial_value:
        The initial value of the resultant array. This should be an array-like object.
    :returns:
        A list of the form [function(array[0]), function(array[1]), ...].
    """
    # We run the first few iterations serially to catch bugs
    front = []
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]

    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]

    # Assemble the workers
    pool_type = ThreadPoolExecutor if multithread else ProcessPoolExecutor
    with pool_type(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]

        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True,
            'disable': not show_progress_bar
        }

        # Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs): pass

    out = initial_value
    out.extend(front)

    # Get the results from the futures.
    _add_func = lambda x: out.extend(x) if extend_result else out.append(x)
    for i, future in tqdm(enumerate(futures)):
        try:
            _add_func(future.result())
        except Exception as e:
            _add_func(e)

    return out


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)