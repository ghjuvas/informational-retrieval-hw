'''
Main module for search: argument parsing and run pipeline code.
'''
import argparse
import logging
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
import pickle
import nlp
import compute_index
import statistics
import similarity

logging.basicConfig(level=logging.INFO)


def initialize_parser() -> argparse.ArgumentParser:
    '''
    Initialize the argument parser.
    '''

    parser = argparse.ArgumentParser(description='Code for search')

    # homework 1
    subparsers = parser.add_subparsers(dest='subparser', help='subcommands for different scenario')
    stats = subparsers.add_parser('stats')

    stats.add_argument(
        'path', type=str,
        help='path to directory with corpuses')

    index_mode = stats.add_mutually_exclusive_group()
    index_mode.add_argument(
        '-m', '--matrix', action='store_true',
        help='create index matrix for search'
    )
    index_mode.add_argument(
        '-d','--dictionary', action='store_true',
        help='create index dictionary for search'
    )

    # homework 2
    search = subparsers.add_parser('search')

    search.add_argument(
        'query', type=str,
        nargs='+',
        help='query for search'
    )
    search.add_argument(
        '-p', '--path', type=str,
        help='path to directory with corpuses; default - search on "Friends" corpus')

    return parser


def get_texts(path: str) -> dict:
    '''
    Get texts from files.
    Raise error if path for files is not valid.
    '''
    if not os.path.exists(path):
        raise ValueError('Path to the directory with files is not valid!')

    logging.info('Collecting corpus...')
    texts = {}
    for root, _, files in os.walk(path):
        for name in files:
            # clear of hidden files
            if name[0] != '.':
                path = os.path.join(root, name)
                with open(path, 'r', encoding='utf-8') as f:
                    texts[name] = f.read()

    return texts


def main() -> None:
    '''
    Main function to run indexig pipeline.
    '''

    argument_parser = initialize_parser()

    args = argument_parser.parse_args()

    if args.subparser == 'stats':

        texts = get_texts(args.path)

        logging.info('Processing texts...')
        lemmas = nlp.processing(texts)

        logging.info('Computing index...')
        if args.matrix:
            vectorizer, index = compute_index.matrix_index(CountVectorizer, lemmas)
            statistics.from_matrix(vectorizer, index)
        if args.dictionary:
            index = compute_index.dictionary_index(lemmas)
            statistics.from_dictionary(index)

    if args.subparser == 'search':

        if args.path:
            # get new search corpus
            logging.info('Getting new search corpus...')

            texts = get_texts(args.path)

            logging.info('Processing texts...')
            lemmas = nlp.processing(texts)

            docs = list(lemmas)

            vectorizer, index = compute_index.matrix_index(TfidfVectorizer, lemmas)
            with open('docs_list.txt', 'w', encoding='utf-8') as fs:
                fs.write('\n'.join(docs))
            with open('vectorizer.pkl', 'wb') as fv:
                pickle.dump(vectorizer, fv)
            sparse.save_npz('index.npz', index)

        else:
            # read precomputed
            logging.info('Reading precomputed index...')
            if not os.path.exists('docs_list.txt'):
                raise ValueError('Path to docs list file is not valid!')
            if not os.path.exists('vectorizer.pkl'):
                raise ValueError('Path to vectorizer file is not valid!')
            if not os.path.exists('index.npz'):
                raise ValueError('Path to index file is not valid!')

            with open('docs_list.txt', 'r', encoding='utf-8') as fs:
                docs = fs.read().split('\n')
            with open('vectorizer.pkl', 'rb') as fv:
                vectorizer = pickle.load(fv)
            index = sparse.load_npz("index.npz")

        logging.info('Searching for similarities...')
        query = ' '.join(args.query)  # read as list because nargs='+' (>=1)
        query_vector = similarity.get_query_vector(query, vectorizer)
        similarity.compute_similarity(query_vector, index, docs)

    logging.info('End of work.')


if __name__ == '__main__':

    main()
