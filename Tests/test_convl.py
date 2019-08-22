import pickle

import spacy
from gensim.models import Word2Vec

'''
def to_process(docs):
    new_docs = []
    nlp = spacy.load('en_core_web_sm')

    for text in docs:

        # Tokenizing & lemmatization
        tokens = []
        doc = nlp(text)
        for word in doc:
            if word.lemma_ != '-PRON-':
                tokens.append(word.lemma_)

        new_docs.append(tokens)

    return new_docs

with open('Datasets/dataset_books', 'rb') as fp:
    b = pickle.load(fp)

with open('Datasets/dataset_electronics', 'rb') as fp:
    e = pickle.load(fp)

with open('Datasets/dataset_dvd', 'rb') as fp:
    d = pickle.load(fp)

with open('Datasets/dataset_kitchen', 'rb') as fp:
    k = dataset_target = pickle.load(fp)

data = b.docs + e.docs
data = data + d.docs
data = data + k.docs

data = to_process(data)

model = Word2Vec(sentences=data, size=200, window=10, min_count=2, workers=10, iter=10)

model.save('Datasets/emb.model')

'''

model = Word2Vec.load('Datasets/emb.model')

