# Informational Retrieval Homework 2022
This repository is for code for homeworks on the course "Informational Retrieval" (year 2022)

## Main information
* Language: Python3
* Needed libraries: run `pip install -r requirements.txt` in the command line

## Homework #1
* Topic: **Inverted index**
* Libraries: `argparse`, `logging`, `os`, `collections`, `string`, `typing` - standard library; `nltk`, `stanza`, `numpy`, `scikit-learn`
* Program code: `search.py` (main), `nlp.py`, `compute_index.py`, `statistics.py`
* Usage: `python3 search.py -d friends-data` - to compute index in the form of a dictionary, `python3 search.py -m friends-data` - to compute index in the form of a matrix

### Documentation
1. **search**
* search.**initialize_parser**() - initialize argument parser.
* search.**main**() - run pipeline for the task: text processing, computing index and getting statistics.

2. **nlp**
* nlp.**processing**(texts) - processing texts with `stanza` - get lemmas of the texts, clear of stopwords and punctuation.

3. **compute_index**
* compute_index.**matrix_index**(lemmas) - get inverted index in the form of a matrix with `scikit-learn`; returns CountVectorizer and index.
* compute_index.**dictionary_index**(lemmas) - get inverted index in the form of a dictionary.

4. **statistics**
* statistics.**from_matrix**(vectorizer, index) - get statistics on the "Friends" corpus (specific) from matrix index with `numpy`.
* statistics.**from_dictionary**(index) - get statistics on the "Friends" corpus (specific) from dictionary index.