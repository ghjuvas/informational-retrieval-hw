'''
Module for computing index by mode.
'''
from collections import defaultdict
from typing import Tuple, Union
import numpy as np
from scipy.sparse._csr import csr_matrix
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch

# load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

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


def bm25_matrix_index(lemmas: list) -> Tuple[CountVectorizer, csr_matrix]:
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


def bert_matrix_index(texts: Union[list, str]):
    '''
    Compute inverted index by BERT encoder in the form of a matrix.
    '''

    # tokenize sentences (without preprocessing)
    encoded_input = tokenizer(
        texts, padding=True,
        truncation=True, max_length=24,
        return_tensors='pt')

    # compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # pooling
    token_embeddings = model_output[0] # first element of model_output contains all token embeddings
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sentence_embeddings = sum_embeddings / sum_mask
    sentence_embeddings = sentence_embeddings.detach().cpu().numpy()  # to array

    return sentence_embeddings


def bert_index_from_batches(texts: list):
    '''
    Get encoded representations of the parts of the texts by BERT
    and concatenate them to full numpy array.
    '''
    batch_size = 50

    res = []
    i = 0
    while i < len(texts):
        part = texts[i:i+batch_size]
        batch = bert_matrix_index(part)
        res.append(batch)
        i += batch_size

    matrix = np.concatenate(res, axis=0)

    return matrix


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