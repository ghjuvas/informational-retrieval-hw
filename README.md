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
* Usage: `python3 search.py search -p friends-data -tfidf книжный шкаф` - to compute index by TF-iDF measure from corpus and then search by query "книжный шкаф" or `python3 search.py search книжный шкаф` - to search by query "книжный шкаф"
* **Important!** Search from precomputed index is performed using the following files (must exist in the current folder): `index.npz` - precomputed index, `vectorizer.pkl` - fitted vectorizer on the data, `docs_list.txt` - list of the names of the documents.

## Homework #3
* Topic: **Okapi BM25**
* Libraries: `argparse`, `logging`, `os`, `collections`, `string`, `typing`, `pickle` - standard library; `nltk`, `stanza`, `numpy`, `scikit-learn`, `scipy`
* Program code: `search.py` (main), `nlp.py`, `compute_index.py`, `similarity.py`
* Usage: `python3 search.py search -f love-corpus/data.jsonl -bm25 книжный шкаф` - to compute index by BM25 measure from corpus file and then search by query "книжный шкаф" or `python3 search.py search -d love-corpus -bm25 книжный шкаф` - to compute index by BM25 measure from corpus files from the directory and then search by query "книжный шкаф" or `python3 search.py search книжный шкаф` - to search by query "книжный шкаф"
* **Important!** Search from precomputed index is performed using the following files (must exist in the current folder): `index.npz` - precomputed index, `vectorizer.pkl` - fitted vectorizer on the data, `docs_list.txt` - list of the names of the documents.
* **Important!** We use 50.000 texts to create an index to search, their morphological and syntactical preprocessing can take time.

## Homework #4
* Topic: **BERT embeddings and evaluation**
* Libraries: `argparse`, `logging`, `os`, `collections`, `string`, `typing`, `pickle` - standard library; `nltk`, `stanza`, `numpy`, `scikit-learn`, `scipy`, `torch`, `transformers`
* Program code: `search.py` (main), `nlp.py`, `compute_index.py`, `similarity.py`, `evaluation.py`
* Usage: `python3 search.py evaluation -f love-corpus/data.jsonl` - to run computing of the indexes and evaluation, `python3 search.py evaluation -d questions.txt answers.txt lem_questions.txt lem_answers.txt`, `python3 search.py evaluation -p bert_index.npy bm25_index.npz vectorizer.pkl docs_list.txt` - to run evaluation on the precomputed indexes.
* **Important!** We use 50.000 texts to create an index to search, their morphological and syntactical preprocessing can take time and encoding by BERT can take time.
* **Additional** For your convenience, we also save texts and parses of the documents in the separated files on your computer. You can use arguments to run the evaluation from the raw and preprocessed texts.
* **How to get the test data?** By [this path](https://drive.google.com/drive/folders/1cMjROa4YyDQbYJEFZDAMyB5FqtW6-Cnb?usp=sharing) you can get all the needed data.

### Documentation
1. **search**
* search.**initialize_parser**() - initialize argument parser.
* search.**parse_data**(path) - parse data from files by extension (text or JSON lines, other - error).
* search.**get_texts**(file, directory) - get texts by mode - from file or files from directory.
* search.**main**() - run pipeline for the task: text processing, computing index and getting statistics.

2. **nlp**
* nlp.**processing**(text) - processing text with `stanza` - get lemmas of the text, clear of stopwords and punctuation.
* nlp.**get_value_lemmas**(texts, mode) - returns list of lemmas of the values of dict or dict vith lemmatized values.

3. **compute_index**
* compute_index.**bm25_matrix_index**(lemmas) - get inverted index in the form of a matrix by BM25 measure with `scikit-learn`; returns vectorizer fitted on the data and computed index.
* compute_index.**bert_matrix_index**(texts) - get encoded representations of the texts by BERT encoder.
* compute_index.**bert_index_from_batches**(texts) - get encoded representations of the texts by BERT encoder by batches of the data (version without memory overflowing).
* compute_index.**matrix_index**(lemmas) - get inverted index in the form of a matrix with `scikit-learn`; returns vectorizer fitted on the data and computed index.
* compute_index.**dictionary_index**(lemmas) - get inverted index in the form of a dictionary.

4. **statistics**
* statistics.**from_matrix**(vectorizer, index) - get statistics on the "Friends" corpus (specific) from matrix index with `numpy`.
* statistics.**from_dictionary**(index) - get statistics on the "Friends" corpus (specific) from dictionary index.
* statistics.**run**() - run statistics pipeline (parsing and processing texts, get statistics by index mode).

5. **similarity**
* similarity.**get_query_vector**(query, vectorizer) - get query vector by data from vectorizer.
* similarity.**compute_cosine_similarity**(query_vector, index) - compute cosine similarity between index and query.
* similarity.**compute_dot_product**(query_vector, index) - compute dot product between index and query.
* similarity.**sort_scores**(similarities, docs) - sort scores and print documents that are most compatible with the query.

6. **evaluation**
* evaluation.**get_accuracy_on_5** - get results on the accuracy metric of the search by top-5 results.
* evaluation.**run**(texts, documents, paths) - run the evaluation pipeline on the Okapi BM25 and BERT indexes by the mode (computing and save indexes, search, and get results of the accuracy metric depending of the arguments).
