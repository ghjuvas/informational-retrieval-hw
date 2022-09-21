# Informational Retrieval Homework 2022
This repository is for code for homeworks on the course "Informational Retrieval" (year 2022)

## Main information
* Language: Python3
* Needed libraries: run `pip install -r requirements.txt` in the command line

## Homework #1
* Topic: **Inverted index**
* Libraries: `argparse`, `logging`, `os`, `collections`, `string`, `typing` - standard library; `nltk`, `stanza`, `numpy`, `scikit-learn`
* Program code: `search.py` (main), `nlp.py`, `compute_index.py`, `statistics.py`
* Usage: `python3 search.py stats -d friends-data` - to compute index in the form of a dictionary, `python3 search.py stats -m friends-data` - to compute index in the form of a matrix

## Homework #2
* Topic: **TF-IDF and similarity of documents**
* Libraries: `argparse`, `logging`, `os`, `collections`, `string`, `typing`, `pickle` - standard library; `nltk`, `stanza`, `numpy`, `scikit-learn`, `scipy`
* Program code: `search.py` (main), `nlp.py`, `compute_index.py`, `similarity.py`
* Usage: `python3 search.py -p friends-data книжный шкаф` - to compute index from corpus and then search by query "книжный шкаф" or `python3 search.py книжный шкаф` - to search by query "книжный шкаф"
* **Important!** Search from precomputed index is performed using the following files (must exist in the current folder): `index.npz` - precomputed index, `vectorizer.pkl` - fitted vectorizer on the data, `docs_list.txt` - list of names of the documents.

### Documentation
1. **search**
* search.**initialize_parser**() - initialize argument parser.
* search.**get_texts**(path) - get texts from files.
* search.**main**() - run pipeline for the task: text processing, computing index and getting statistics.

2. **nlp**
* nlp.**processing**(texts) - processing texts with `stanza` - get lemmas of the texts, clear of stopwords and punctuation.

3. **compute_index**
* compute_index.**matrix_index**(lemmas) - get inverted index in the form of a matrix with `scikit-learn`; returns vectorizer fitted on the data and icomputed index.
* compute_index.**dictionary_index**(lemmas) - get inverted index in the form of a dictionary.

4. **statistics**
* statistics.**from_matrix**(vectorizer, index) - get statistics on the "Friends" corpus (specific) from matrix index with `numpy`.
* statistics.**from_dictionary**(index) - get statistics on the "Friends" corpus (specific) from dictionary index.

5. **similarity**
* similarity.**get_query_vector**(query, vectorizer) - get query vector by data from vectorizer.
* similarity.**compute_similarity**(query_vector, index, docs) - print documents that are most compatible with the query.