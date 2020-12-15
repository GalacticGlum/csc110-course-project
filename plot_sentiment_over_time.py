"""Plot sentiment of the given words over time."""
import random
import argparse
import tensorflow as tf
from typing import List
from pathlib import Path
from dateutil.parser import parse

import tikzplotlib
import matplotlib.pyplot as plt

from logger import logger
from temporal_analysis import get_sentiment_over_time

def sample_lines(filename: str, k: int) -> List[str]:
    """Randomly sample k lines from the given file."""
    sample = []
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()
        indices = sorted(random.sample(range(filesize), k))
        for i in range(k):
            f.seek(indices[i])
            f.readline()
            sample.append(f.readline().rstrip())

    return sample


def main(args: argparse.Namespace) -> None:
    """Main entrypoint for the script."""
    files = Path(args.data_root).glob(args.glob)
    temporal_tweets = []
    for file in files:
        filename = file.stem
        filename = filename[filename.find('-') + 1:filename.find('_p')].replace('_', '-')
        file_date = parse(filename, fuzzy=True).replace(day=1)

        # Get tweets (1K per month to run sentiment analysis)
        tweets = sample_lines(file, k=1000)
        temporal_tweets.append((file_date, tweets))

    model = tf.saved_model.load(str(args.checkpoint))
    sentiments = get_sentiment_over_time(temporal_tweets, model=model)

    figsize = (args.figure_width / args.figure_dpi, args.figure_height / args.figure_dpi)
    plt.figure(figsize=figsize, dpi=args.figure_dpi)
    # Plot data
    x = [date for date, _ in sentiments]
    y = [sentiment for _, sentiment in sentiments]

    plt.xlabel('Date')
    plt.ylabel('Average Climate Change Sentiment')
    plt.plot_date(x, y, '-o')

    if not args.output_path:
        plt.show()
    else:
        output_format = (args.output_path.suffix or 'png').replace('.', '')
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == 'tex' or output_format == 'latex':
            tikzplotlib.save(args.output_path)
        else:
            plt.savefig(args.output_path, dpi=args.export_dpi)
        logger.info('Exported figure to {}'.format(args.output_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualisation tools for temporal analysis.')
    parser.add_argument('checkpoint', type=Path, help='The folder containing the model checkpoint.')
    parser.add_argument('data_root', type=Path, help='The folder containing the corpus files.')
    parser.add_argument('--glob', type=str, default='20*.txt',
                        help='Pattern to match checkpoint folders. Defaults to \'00001-20*\'')
    parser.add_argument('-o', '--output', dest='output_path', type=Path, default=None,
                        help='The file to write the figure to.')
    parser.add_argument('-fw', '--figure-width', type=int, default=800,
                        help='The width of the exported file.')
    parser.add_argument('-fh', '--figure-height', type=int, default=600,
                        help='The heght of the exported file.')
    parser.add_argument('-dpi', '--figure-dpi', type=int, default=96,
                        help='The DPI of the exported file.')
    parser.add_argument('-edpi', '--export-dpi', type=int, default=96,
                        help='The DPI of the exported file.')
    main(parser.parse_args())
