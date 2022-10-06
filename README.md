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
* Usage: `python3 search.py search -p friends-data книжный шкаф` - to compute index from corpus and then search by query "книжный шкаф" or `python3 search.py search книжный шкаф` - to search by query "книжный шкаф"
* **Important!** Search from precomputed index is performed using the following files (must exist in the current folder): `index.npz` - precomputed index, `vectorizer.pkl` - fitted vectorizer on the data, `docs_list.txt` - list of the names of the documents.

## Homework #3
* Topic: **Okapi BM25**
* Libraries: `argparse`, `logging`, `os`, `collections`, `string`, `typing`, `pickle` - standard library; `nltk`, `stanza`, `numpy`, `scikit-learn`, `scipy`
* Program code: `search.py` (main), `nlp.py`, `compute_index.py`, `similarity.py`
* Usage: `python3 search.py search -f love-corpus/data.jsonl -b книжный шкаф` - to compute index by BM25 measure from corpus file and then search by query "книжный шкаф" or `python3 search.py search -d love-corpus -b книжный шкаф` - to compute index by BM25 measure from corpus files from the directory and then search by query "книжный шкаф" or `python3 search.py search книжный шкаф` - to search by query "книжный шкаф"
* **Important!** Search from precomputed index is performed using the following files (must exist in the current folder): `index.npz` - precomputed index, `vectorizer.pkl` - fitted vectorizer on the data, `docs_list.txt` - list of the names of the documents.
* **Important!** We use 50.000 texts to create an index to search, their morphological and syntactical preprocessing can take time.

### Documentation
1. **search**
* search.**initialize_parser**() - initialize argument parser.
* search.**parse_data**() - parse data from files by extension (text or JSON lines, other - error).
* search.**get_texts**(path) - get texts by mode - from file or files from directory.
* search.**main**() - run pipeline for the task: text processing, computing index and getting statistics.

2. **nlp**
* nlp.**processing**(texts) - processing texts with `stanza` - get lemmas of the texts, clear of stopwords and punctuation.

3. **compute_index**
* compute_index.**matrix_index**(lemmas) - get inverted index in the form of a matrix by BM25 measure with `scikit-learn`; returns vectorizer fitted on the data and computed index.
* compute_index.**matrix_index**(lemmas) - get inverted index in the form of a matrix with `scikit-learn`; returns vectorizer fitted on the data and computed index.
* compute_index.**dictionary_index**(lemmas) - get inverted index in the form of a dictionary.

4. **statistics**
* statistics.**from_matrix**(vectorizer, index) - get statistics on the "Friends" corpus (specific) from matrix index with `numpy`.
* statistics.**from_dictionary**(index) - get statistics on the "Friends" corpus (specific) from dictionary index.

5. **similarity**
* similarity.**get_query_vector**(query, vectorizer) - get query vector by data from vectorizer.
* similarity.**compute_cosine_similarity**(query_vector, index) - compute cosine similarity between index and query.
* similarity.**compute_dot_product**(query_vector, index) - compute dot product between index and query.
* similarity.**sort_scores**(similarities, docs) - sort scores and print documents that are most compatible with the query.