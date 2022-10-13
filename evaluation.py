'''
Module for running evaluation between two search indexes (by Okapi BM25 and BERT encoder)
on the Love corpus (from Mail.ru).
'''
import logging
import pickle
from typing import Optional
from scipy import sparse
import tqdm
import numpy as np
import compute_index
import similarity
import nlp

logging.basicConfig(level=logging.INFO)


def get_accuracy_on_5(similarities: np.ndarray, docs: list) -> float:
    sorted_scores, docs_array = similarity.sort_scores(similarities, docs)
    texts_matrix = docs_array[sorted_scores][:5, :]
    accuracy = len(texts_matrix[texts_matrix == docs_array]) / len(docs_array)
    return accuracy


def run(texts: Optional[dict], documents: Optional[list], precomputed: Optional[list]) -> None:

    if texts or documents:

        if texts:

            docs = list(texts)  # questions
            with open('questions.txt', 'w', encoding='utf-8') as fq:
                fq.write('\n'.join(docs))

            # bm25
            logging.info('Preprocessing answers...')
            lemmatized_answers = nlp.get_value_lemmas(texts, 'list')
            with open('lem_answers.txt', 'w', encoding='utf-8') as fl:
                fl.write('\n'.join(lemmatized_answers))

            logging.info('Preprocessing documents...')
            lemmatized_docs = [' '.join(nlp.processing(text)) for text in tqdm.tqdm(docs)]
            with open('lem_questions.txt', 'w', encoding='utf-8') as flq:
                flq.write('\n'.join(lemmatized_docs))

            # bert
            answers = list(texts.values())
            with open('answers.txt', 'w', encoding='utf-8') as fa:
                fa.write('\n'.join(answers))

        if documents:

            # read raw and lemmatized corpus
            with open(documents[0], 'r', encoding='utf-8') as fq:
                docs = fq.read().split('\n')
            with open(documents[1], 'r', encoding='utf-8') as fa:
                answers = fa.read().split('\n')
            with open(documents[2], 'r', encoding='utf-8') as flq:
                lemmatized_docs = flq.read().split('\n')
            with open(documents[3], 'r', encoding='utf-8') as fla:
                lemmatized_answers = fla.read().split('\n')

        logging.info('Computing BM25 index...')
        vectorizer, bm25_index = compute_index.bm25_matrix_index(lemmatized_answers)

        logging.info('Computing BERT index...')
        bert_index = compute_index.bert_index_from_batches(answers)
        docs_bert_vectors = compute_index.bert_index_from_batches(docs)

        # save indexes
        with open('vectorizer.pkl', 'wb') as fv:
            pickle.dump(vectorizer, fv)
        sparse.save_npz("bm25_index.npz", bm25_index)
        np.save('bert_index.npy', bert_index)
        np.save('bert_query_index.npy', docs_bert_vectors)

    if precomputed:

        # read precomputed
        bm25_index = sparse.load_npz(precomputed[0])
        with open(precomputed[1], 'rb') as fv:
            vectorizer = pickle.load(fv)
        bert_index = np.load(precomputed[2])
        docs_bert_vectors = np.load(precomputed[3])
        with open(precomputed[4], 'r', encoding='utf-8') as fs:
            docs = fs.read().split('\n')

        logging.info('Preprocessing documents...')
        lemmatized_docs = [' '.join(nlp.processing(text)) for text in tqdm.tqdm(docs)]

    logging.info('Computing BM25 similarities...')
    docs_count_vectors = vectorizer.transform(lemmatized_docs)
    bm25_similarities = similarity.compute_dot_product(bm25_index, docs_count_vectors)
    bm25_accuracy = get_accuracy_on_5(bm25_similarities, docs)

    logging.info('Computing BERT similarities...')
    bert_similarities = similarity.compute_cosine_similarity(bert_index, docs_bert_vectors)
    bert_accuracy = get_accuracy_on_5(bert_similarities, docs)

    print('Accuracy on top-5 by BM25:', bm25_accuracy)
    print('Accuracy on top-5 by BERT:', bert_accuracy)