from typing import Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_query_vector(query: str, vectorizer: Union[CountVectorizer, TfidfVectorizer]) -> np.array:
    '''
    Get vector of the query.
    '''
    query_vector = vectorizer.transform([query])

    return query_vector


def compute_similarity(query_vector: np.array, index: np.array, docs: list) -> None:
    '''
    Compute similarity between query and documents.
    '''
    similarities = list(cosine_similarity(query_vector, index)[0])
    sim_dict = {k: v for k, v in zip(docs, similarities)}
    documents = list(dict(sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)))

    for doc in documents:
        print(doc)