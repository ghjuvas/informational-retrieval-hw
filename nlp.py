'''
Module for morphological and syntactical parsing of texts.
In the end, we get a set of lemmas according to the documents.
'''
import logging
from string import punctuation
from nltk.corpus import stopwords
import stanza

logging.getLogger('stanza').disabled = True

PUNCTUATION = punctuation + '«»—'
RUS_SW = set(stopwords.words('russian'))


def processing(text: str) -> dict:
    '''
    Processing one text with stanza (https://stanfordnlp.github.io/stanza/).
    Clear of stopwords and punctuation. Get lemmas
    '''
    # if russian package is not downloaded
    # it will be downloaded automatically
    nlp = stanza.Pipeline(lang='ru', processors='tokenize,pos,lemma')

    lemmas = []
    doc = nlp(text)
    for sent in doc.sentences:
        for word in sent.words:
            lemma = word.lemma.lower()
            if lemma not in RUS_SW and lemma not in PUNCTUATION:
                lemmas.append(lemma)

    return lemmas