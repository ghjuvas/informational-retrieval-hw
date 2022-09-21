'''
Module for computing index by mode.
'''
from collections import defaultdict
from typing import Tuple, Union
from scipy.sparse._csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def matrix_index(vectorizer: Union[CountVectorizer, TfidfVectorizer],
    lemmas: dict) -> Tuple[CountVectorizer, csr_matrix]:
    '''
    Compute inverted index in the form of a matrix.
    '''

    # prepare data
    lemmas = [' '.join(text_lemmas) for text_lemmas in list(lemmas.values())]

    # transform to index
    vectorizer = vectorizer(analyzer='word')
    X = vectorizer.fit_transform(lemmas)

    return vectorizer, X


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