from typing import Union
import numpy as np
from scipy.sparse._csr import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_query_vector(query: str, vectorizer: Union[CountVectorizer, TfidfVectorizer]) -> csr_matrix:
    '''
    Get vector of the query.
    '''
    query_vector = vectorizer.transform([query])

    return query_vector


def compute_cosine_similarity(query_vector: Union[csr_matrix, np.ndarray],
                              index: Union[csr_matrix, np.ndarray]) -> None:
    '''
    Compute similarity between query and documents by cosine similarity.
    '''
    similarities = cosine_similarity(index, query_vector)

    return similarities


def compute_dot_product(query_vector: Union[csr_matrix, np.ndarray],
                        index: Union[csr_matrix, np.ndarray]):
    '''
    Compute similarity between query and documents by dot product.
    '''
    similarities = np.dot(index, query_vector.T)

    return similarities


def sort_scores(similarities: Union[csr_matrix, np.ndarray], docs: list):
    '''
    Sort scores after computing similarities.
    '''
    if not isinstance(similarities, np.ndarray):
        similarities = similarities.toarray()
    sorted_scores = np.argsort(similarities, axis=0)[::-1]  # arg -> index
    docs = np.array(docs)

    for idx, doc in enumerate(list(docs[sorted_scores.ravel()])):
        print(f'{idx}.', doc)