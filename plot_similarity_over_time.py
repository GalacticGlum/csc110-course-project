"""Plot similarity of the given words over time."""
import argparse
from pathlib import Path
from dateutil.parser import parse
from datetime import date, datetime

import tikzplotlib
import matplotlib.pyplot as plt

from logger import logger
from word_embeddings import WordEmbeddings
from temporal_analysis import plot_similarity_over_time


def main(args: argparse.Namespace) -> None:
    """Main entrypoint for the script."""
    temporal_embeddings = []
    paths = Path(args.checkpoint_root).glob(args.glob)
    for path in paths:
        path_name = path.stem
        path_name = path_name[path_name.find('-') + 1:path_name.find('_p')].replace('_', '-')
        path_date = parse(path_name, fuzzy=True).replace(day=1)

        # Load the word embeddings
        embeddings = WordEmbeddings(
            checkpoint_filepath=path,
            suffix_tree=False,
            nearest_neighbours=False
        )
        temporal_embeddings.append((path_date.date(), embeddings))

    figsize = (args.figure_width / args.figure_dpi, args.figure_height / args.figure_dpi)
    plt.figure(figsize=figsize, dpi=args.figure_dpi)

    # Draw the graph.
    plot_similarity_over_time(temporal_embeddings, args.word_a, args.word_b)

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
    parser.add_argument('checkpoint_root', type=Path, help='The root folder containing all checkpoints.')
    parser.add_argument('word_a', type=str, help='The first word to compare.')
    parser.add_argument('word_b', type=str, help='The second word to compare.')
    parser.add_argument('--glob', type=str, default='00001-20*',
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