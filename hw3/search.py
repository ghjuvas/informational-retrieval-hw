'''
Main module for search: argument parsing and run pipeline code.
'''
import argparse
import logging
import os
import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
import tqdm
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
        help='path to the directory with corpuses')

    index_mode_stats = stats.add_mutually_exclusive_group()
    index_mode_stats.add_argument(
        '-m', '--matrix', action='store_true',
        help='create index matrix for search'
    )
    index_mode_stats.add_argument(
        '-d','--dictionary', action='store_true',
        help='create index dictionary for search'
    )

    # homework 2 & 3
    search = subparsers.add_parser('search')

    search.add_argument(
        'query', type=str,
        nargs='+',
        help='query for search'
    )

    corpus_mode = search.add_mutually_exclusive_group()
    corpus_mode.add_argument(
        '-f', '--file', type=str,
        help='path to the file with corpus')
    corpus_mode.add_argument(
        '-d', '--directory', type=str,
        help='path to the directory with corpuses')

    index_mode_search = search.add_mutually_exclusive_group()
    index_mode_search.add_argument(
        '-t', '--tfidf', action='store_true',
        help='create index matrix by TF-IDF for search'
    )
    index_mode_search.add_argument(
        '-b','--bm25', action='store_true',
        help='create index matrix by BM25 for search'
    )

    return parser


def parse_data(path: str):
    '''
    Parse data from files by format (text or JSON lines)
    '''
    ext = os.path.splitext(path)[1]
    # print(ext)

    if ext == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            texts = f.read()

    elif ext == '.jsonl':

        texts = {}

        with open(path, 'r', encoding='utf-8') as f:
            while len(texts) < 50000:

                try:
                    d = json.loads(next(f))
                    max_value = -1  # if value of the author is 0
                    text = None
                    answers = d.get('answers', None)
                    if isinstance(answers, list):
                        # parse from the specified format of the Mail corpus
                        for answer in answers:
                            rating = answer.get('author_rating', None)
                            if rating:
                                val = rating.get('value', None)
                                if val:
                                    new_value = int(val)
                                    if new_value > max_value:
                                        max_value = new_value
                                        text = answer.get('text', None)
                    if text is not None:  # some questions do not have answers
                        texts[text] = text

                except StopIteration:  # handle iterator`s stop
                    logging.info('Iteration stopped earlier than 50000!')
                    return texts

    else:
        raise ValueError('Program does not support reading this file format!')

    return texts


def get_texts(path: str, mode: str) -> dict:
    '''
    Get texts from files.
    Raise error if path for files is not valid.
    '''
    if not os.path.exists(path):
        raise ValueError('Path to the directory with files is not valid!')

    logging.info('Collecting corpus...')

    if mode == 'directory':

        texts = {}
        for root, _, files in os.walk(path):
            for name in files:
                # clear of hidden files
                if name[0] != '.':
                    path = os.path.join(root, name)
                    data = parse_data(path)
                    if os.path.splitext(path)[1] == '.jsonl':
                        # one document - one full text
                        texts[name] = ' '.join(list(data.values()))
                    else:
                        texts[name] = data

    if mode == 'file':

        texts = parse_data(path)

    return texts


def main() -> None:
    '''
    Main function to run indexing pipeline.
    '''

    argument_parser = initialize_parser()

    args = argument_parser.parse_args()

    if args.subparser == 'stats':

        if args.file:
            texts = get_texts(args.file, 'file')
        if args.directory:
            texts = get_texts(args.directory, 'directory')

        logging.info('Processing texts...')
        lemmas = {}
        for index, text in texts.items():
            text_lemmas = nlp.processing(text)
            lemmas[index] = text_lemmas  # matrix (full text) or dictionary (lemmas)

        logging.info('Computing index...')
        if args.matrix:
            lemmas = [' '.join(text) for text in lemmas.values()]
            vectorizer, index = compute_index.matrix_index(CountVectorizer, lemmas)
            statistics.from_matrix(vectorizer, index)
        if args.dictionary:
            index = compute_index.dictionary_index(lemmas)
            statistics.from_dictionary(index)

    if args.subparser == 'search':

        if args.file or args.directory:

            # get new search corpus
            logging.info('Getting new search corpus...')

            if args.file:
                texts = get_texts(args.file, 'file')
                # print(len(texts))
            if args.directory:
                texts = get_texts(args.directory, 'directory')

            logging.info('Processing texts...')
            lemmas = []
            for text in tqdm.tqdm(list(texts.values())):
                lemmatized = ' '.join(nlp.processing(text))
                lemmas.append(lemmatized)

            with open('lemmatized.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(lemmas))

            docs = list(texts)

            # compute index
            if args.tfidf:
                vectorizer, index = compute_index.matrix_index(TfidfVectorizer, lemmas)
            if args.bm25:
                vectorizer, index = compute_index.bm25_matrix_index(lemmas)
            else:
                raise ValueError('Choose the mode of index creation!')

            # save files
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

        # similarities
        logging.info('Searching for similarities...')
        query = ' '.join(args.query)  # read as list because nargs='+' (>=1)
        query = ' '.join(nlp.processing(query))
        query_vector = similarity.get_query_vector(query, vectorizer)
        if args.tfidf:
            similarities = similarity.compute_cosine_similarity(query_vector, index)
        if args.bm25:
            similarities = similarity.compute_dot_product(query_vector, index)
        similarity.sort_scores(similarities, docs)

    logging.info('End of work.')


if __name__ == '__main__':

    main()
