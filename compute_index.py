'''
Module for computing index by mode.
'''
from collections import defaultdict
from typing import Tuple, Union
import numpy as np
from scipy.sparse._csr import csr_matrix
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

K = 2
B = 0.75


def matrix_index(vectorizer: Union[CountVectorizer, TfidfVectorizer],
    lemmas: list) -> Tuple[CountVectorizer, csr_matrix]:
    '''
    Compute inverted index in the form of a matrix.
    '''

    # transform to index
    vectorizer = vectorizer(analyzer='word')
    X = vectorizer.fit_transform(lemmas)

    return vectorizer, X


def bm25_matrix_index(lemmas: list) -> Tuple[CountVectorizer]:
    '''
    Compute index by BM25 measure.
    '''

    # tf & query vectorization
    count_vectorizer, tf = matrix_index(CountVectorizer, lemmas)

    # idf
    tfidf_vectorizer, _ = matrix_index(TfidfVectorizer, lemmas)

    idf = tfidf_vectorizer.idf_
    idf = np.expand_dims(idf, axis=0)

    # documents measures
    len_d = tf.sum(axis=1)
    avgdl = len_d.mean()

    data = []
    for i, j in zip(*tf.nonzero()):
        # sparse[row,column] - access to the particular element
        numerator = idf[0][j] * tf[i,j] * (K + 1)
        denominator = tf[i,j] + (K * (1 - B + B * len_d[i,0] / avgdl))
        data.append(numerator / denominator)

    rows = list(tf.nonzero()[0])
    columns = list(tf.nonzero()[1])

    index = sparse.csr_matrix((data, (rows, columns)), shape=tf.shape)

    return count_vectorizer, index


def dictionary_index(lemmas: dict) -> defaultdict:
    '''
    Compute inverted index in the form of a dictionary.
    '''

    # list like posting list
    # save information about frequency of occurence
    index_dictionary = defaultdict(list)
    for index, text in lemmas.items():
        for lemma in text:
            index_dictionary[lemma].append(index)

    return index_dictionary