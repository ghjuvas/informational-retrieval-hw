'''
Main module for search: argument parsing and run pipeline code.
'''
import argparse
import logging
import os
import json
import pickle
from typing import Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import tqdm
import nlp
import compute_index
import statistics
import similarity
import evaluation

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
        '-tfidf', action='store_true',
        help='create index matrix by TF-IDF for search'
    )
    index_mode_search.add_argument(
        '-bm25', action='store_true',
        help='create index matrix by BM25 for search'
    )
    # homework 4 (task 1)
    index_mode_search.add_argument(
        '-bert', action='store_true',
        help='create index matrix by BERT encoder for search'
    )

    # homework 4 (task 2)
    evaluation = subparsers.add_parser('evaluation')
    paths = evaluation.add_mutually_exclusive_group()

    paths.add_argument(
        '-f', '--file', type=str,
        help='path to the file with corpus'
    )
    paths.add_argument(
        '-d', '--documents', nargs=4,
        help='paths of documents (1st), values (2nd), lemmatized documents (3rd), lemmatized values (4th)'
    )
    paths.add_argument(
        '-p', '--precomputed', nargs=5,
        help='paths of BM25 index (1st), count vectorizer (2nd), BERT index (3rd), VERT query index (5th), and documents list (4th)'
    )


    return parser


def parse_data(path: str) -> Union[str, dict]:
    '''
    Parse data from files by format (text or JSON lines)
    '''
    ext = os.path.splitext(path)[1]

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
                    question = d.get('question', None)
                    if question:
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
                        if text:  # some questions do not have answers
                            texts[question] = text

                except StopIteration:  # handle iterator`s stop
                    logging.info('Iteration stopped earlier than 50000!')

    else:
        raise ValueError('Program does not support reading this file format!')

    return texts


def get_texts(file: Optional[str], directory: Optional[str]) -> dict:
    '''
    Get texts from files.
    Raise error if path for files is not valid.
    '''

    if directory:
        if not os.path.exists(directory):
            raise ValueError('Path to the corpus is not valid!')
        logging.info('Collecting corpus...')
        texts = {}
        for root, _, files in os.walk(directory):
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

    if file:
        if not os.path.exists(file):
            raise ValueError('Path to the corpus is not valid!')
        logging.info('Collecting corpus...')
        texts = parse_data(file)

    return texts


def main() -> None:
    '''
    Main function to run indexing pipeline.
    '''

    argument_parser = initialize_parser()

    args = argument_parser.parse_args()

    if args.subparser == 'stats':
        # statistics logic is separated

        statistics.run()

    if args.subparser == 'search':
        # main logic of the program

        if args.file or args.directory:
            # then compute index and save files

            # get new search corpus
            logging.info('Getting new search corpus...')

            # different file parsing
            texts = get_texts(args.file, args.directory)

            docs = list(texts)  # 

            if args.bert:
                # without preprocessing
                logging.info('Computing index...')
                index = compute_index.bert_matrix_index(docs)

            elif args.tfidf or args.bm25:

                logging.info('Processing texts...')
                lemmas = nlp.get_value_lemmas(texts, 'list')

                # compute index
                logging.info('Computing index...')
                if args.tfidf:
                    vectorizer, index = compute_index.matrix_index(TfidfVectorizer, lemmas)
                if args.bm25:
                    vectorizer, index = compute_index.bm25_matrix_index(lemmas)
                # save vectorizer after computing
                # as specific operation of this two methods
                with open('vectorizer.pkl', 'wb') as fv:
                    pickle.dump(vectorizer, fv)

            else:
                raise ValueError('Choose the mode of index creation!')

            # save files
            with open('docs_list.txt', 'w', encoding='utf-8') as fs:
                fs.write('\n'.join(docs))
            sparse.save_npz('index.npz', index)

        else:
            # then read precomputed
            logging.info('Reading precomputed index...')

            # path errors
            if not os.path.exists('docs_list.txt'):
                raise ValueError('Path to docs list file is not valid!')
            if not os.path.exists('index.npz'):
                raise ValueError('Path to index file is not valid!')
            if args.tfidf or args.bm25:
                if not os.path.exists('vectorizer.pkl'):
                    raise ValueError('Path to vectorizer file is not valid!')
                with open('vectorizer.pkl', 'rb') as fv:
                    vectorizer = pickle.load(fv)

            # read
            with open('docs_list.txt', 'r', encoding='utf-8') as fs:
                docs = fs.read().split('\n')
            index = sparse.load_npz("index.npz")

        # similarities
        logging.info('Searching for similarities...')
        query = ' '.join(args.query)  # read as list because nargs='+' (>=1)
        query = ' '.join(nlp.processing(query))

        if args.tfidf:
            query_vector = similarity.get_query_vector(query, vectorizer)
            similarities = similarity.compute_cosine_similarity(query_vector, index)
        elif args.bm25:
            query_vector = similarity.get_query_vector(query, vectorizer)
            similarities = similarity.compute_dot_product(query_vector, index)
        elif args.bert:
            query_vector = compute_index.bert_matrix_index(query)
            similarities = similarity.compute_dot_product(query_vector, index)
        else:
            raise ValueError('Choose mode of the index!')

        sorted_scores, docs = similarity.sort_scores(similarities, docs)
        for idx, doc in enumerate(list(docs[sorted_scores.ravel()])):
            print(f'{idx}.', doc)

    if args.subparser == 'evaluation':
        # evaluation logic is separated

        texts = None
        if args.file:
            texts = get_texts(args.file, None)

        evaluation.run(texts, args.documents, args.precomputed)

    logging.info('End of work.')


if __name__ == '__main__':

    main()
