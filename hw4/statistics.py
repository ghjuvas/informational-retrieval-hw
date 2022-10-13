'''
Module for getting statistics.
'''
from collections import Counter
import logging
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import search
import compute_index

logging.basicConfig(level=logging.INFO)

NAMES = {'Моника': ['моника', 'мон'],
'Рэйчел': ['рэйчел', 'рейч'],
'Чендлер': ['чендлер', 'чэндлер', 'чен'],
'Фиби': ['фиби', 'фибс'],
'Росс': ['росс'],
'Джоуи': ['джоуи', 'джои', 'джо']}


def from_matrix(vectorizer: CountVectorizer, index: np.array) -> None:
    '''
    Get statistics on words from matrix.
    '''
    def find_frequent_word(nums, words, num, min_max_condition):
        if min_max_condition:
            # choose columns where 
            columns = np.where(nums == num)
        else:
            columns = np.where(~np.any(nums == 0, axis=0))
            # print(words)
        words = list(words[columns])
        words = ', '.join(words)
        return words

    # prepare data
    # array of words
    words_vector = np.array(vectorizer.get_feature_names())

    # inverted index with words
    index_matrix = index.toarray()

    # frequency matrix with words
    freq_array = np.asarray(index.sum(axis=0)).ravel()

    # statistics
    # most frequent word
    max_freq = str(freq_array.max())
    print('Most frequent words:',
    find_frequent_word(
        freq_array, words_vector, max_freq, min_max_condition=True
        ))

    # less frequent word
    min_freq = str(freq_array.min())
    print('Less frequent words:',
    find_frequent_word(
        freq_array, words_vector, max_freq, min_max_condition=True
        ))

    # words that are in all of the documents
    # column with name must not have zeros
    print('Words that are in all of the documents:',
    find_frequent_word(
        index_matrix, '0', min_max_condition=False
        ))

    # most popular person
    c = Counter()
    for person, names in NAMES.items():
        for name in names:
            column = np.where(words_vector == name)
            value = freq_array[column]
            if value.size > 0:
                freq = int(value)
            else:
                freq = 0
            c[person] += freq
    print('Most popular person:', c.most_common()[0][0])


def from_dictionary(index):
    '''
    Get statistics on words from dictionary.
    '''

    # prepare data
    freq_list = list(index.values())
    length_list = list(map(len, freq_list))
    max_freq = max(length_list)
    min_freq = min(length_list)

    # statistics
    most_frequent = []
    less_frequent = []
    from_all_docs = []
    c = Counter()
    for word, docs in index.items():
        if len(docs) == max_freq:
            most_frequent.append(word)
        if len(docs) == min_freq:
            less_frequent.append(word)
        if len(set(docs)) == 165:  # number of documents in the collection
            from_all_docs.append(word)
        for person, names in NAMES.items():
            if word in names:
                c[person] += len(docs)

    print('Most frequent words:', ', '.join(most_frequent))
    print('Less frequent words:', ', '.join(less_frequent))
    print('Words that are in all of the documents:', ', '.join(from_all_docs))
    print('Most popular person:', c.most_common()[0][0])


def run(path: str, matrix: bool) -> None:
    '''
    Run statistics pipeline.
    '''
    texts = search.get_texts(None, path)

    logging.info('Processing texts...')
    lemmas = search.get_value_lemmas(texts, dict)

    logging.info('Computing index...')
    if matrix:
        lemmas = [' '.join(text) for text in lemmas.values()]
        vectorizer, index = compute_index.matrix_index(CountVectorizer, lemmas)
        from_matrix(vectorizer, index)
    else:
        index = compute_index.dictionary_index(lemmas)
        from_dictionary(index)
