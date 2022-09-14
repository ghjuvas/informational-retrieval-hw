'''
Main module for search: argument parsing and run pipeline code.
'''
import argparse
import logging
import os
import nlp
import compute_index
import statistics

logging.basicConfig(level=logging.INFO)


def initialize_parser() -> argparse.ArgumentParser:
    '''
    Initialize the argument parser.
    '''

    parser = argparse.ArgumentParser(description='Code for search')

    parser.add_argument(
        'path', type=str,
        help='path to directory with corpuses')

    index_mode = parser.add_mutually_exclusive_group()
    index_mode.add_argument(
        '-m', '--matrix', action='store_true',
        help='create index matrix for search'
    )
    index_mode.add_argument(
        '-d','--dictionary', action='store_true',
        help='create index dictionary for search'
    )

    return parser


def main() -> None:
    '''
    Main function to run indexig pipeline.
    '''

    argument_parser = initialize_parser()

    args = argument_parser.parse_args()

    if not os.path.exists(args.path):
        raise ValueError('Path for directory with files is not valid!')

    logging.info('Collecting corpus...')
    texts = {}
    for root, _, files in os.walk('friends-data'):
        for name in files:
            # clear of hidden files
            if name[0] != '.':
                path = os.path.join(root, name)
                with open(path, 'r', encoding='utf-8') as f:
                    texts[name] = f.read()

    logging.info('Processing texts...')
    lemmas = nlp.processing(texts)

    logging.info('Computing index...')
    if args.matrix:
        index, matrix = compute_index.matrix_index(lemmas)
        statistics.from_matrix(index, matrix)
    if args.dictionary:
        index = compute_index.dictionary_index(lemmas)
        statistics.from_dictionary(index)

    logging.info('End of work.')


if __name__ == '__main__':

    main()
