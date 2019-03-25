import random

import gensim
import pickle
import string
import nltk
import numpy as np
from nltk import WordNetLemmatizer
from scipy.misc import comb
from stemming.porter2 import stem
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
punctuation = "[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]:0123456789"

def to_process(docs, pos):

    # Reading stop-words
    stop_words = set(stopwords.words('english'))

    new_docs = []

    for text in docs:

        # Tokenizing
        text = text.lower()
        tokens = nltk.word_tokenize(text)

        result = []

        # Removing stop words
        for word in tokens:
            if word not in stop_words and word not in punctuation:
                result.append(word)

        # lemmatizing
        result = [lemmatizer.lemmatize(word) for word in tokens]

        # POS filter: only adverbs, adjectives and nouns
        pos_tags = nltk.pos_tag(result)
        result_pos = []

        if pos == '3':
            for word in pos_tags:
                if word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'NNP' or \
                        word[1] == 'NNPS':
                    result_pos.append(word[0])

        elif pos == '1':
            for word in pos_tags:
                if word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS':
                    result_pos.append(word[0])

        elif pos == '5':
            for word in pos_tags:
                if word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS' or word[1] == 'RB' or \
                        word[1] == 'RB' or word[1] == 'RBR' or word[1] == 'RBS':
                    result_pos.append(word[0])

        elif pos == '2':
            for word in pos_tags:
                if word[1] == 'RB' or word[1] == 'RBR' or word[1] == 'RBS':
                    result_pos.append(word[0])

        elif pos == '4':
            for word in pos_tags:
                if word[1] == 'VB' or word[1] == 'VBD' or word[1] == 'VBG' or \
                        word[1] == 'VBN' or word[1] == 'VBP' or word[1] == 'VBZ':
                    result_pos.append(word[0])
        elif pos == '6':
            for word in pos_tags:
                if word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS' or word[1] == 'RB' or \
                        word[1] == 'RB' or word[1] == 'RBR' or word[1] == 'RBS' or word[1] == 'NN' or word[1] == 'NNS' \
                        or word[1] == 'NNP' or word[1] == 'NNPS':
                    result_pos.append(word[0])
        else:
            result_pos = result
        new_docs.append(result_pos)

    return new_docs

def training_word2vec():
    with open('Datasets/dataset_kindle_10k', 'rb') as fp:
        target = pickle.load(fp)

    sentences = to_process(target.docs, 6)

    # train word2vec on the two sentences
    model = gensim.models.Word2Vec(sentences, min_count=1)

    return model

def gen_data():
    with open('Datasets/dataset_books', 'rb') as fp:
        data_source_a = pickle.load(fp)
    with open('Datasets/dataset_kitchen', 'rb') as fp:
        data_source_b = pickle.load(fp)
    with open('Datasets/dataset_electronics', 'rb') as fp:
        data_source_c = pickle.load(fp)

    def suffling(data):
        docs = data.docs
        labels = data.labels

        c = list(zip(docs, labels))

        random.shuffle(c)

        docs, labels = zip(*c)

        data.docs = docs
        data.labels = labels

        return data


    data_source_a.docs = to_process(data_source_a.docs, '6')
    data_source_b.docs = to_process(data_source_b.docs, '6')
    data_source_c.docs = to_process(data_source_c.docs, '6')

    data_source_a = suffling(data_source_a)
    data_source_b = suffling(data_source_b)
    data_source_c = suffling(data_source_c)

    print(data_source_a.docs)

    with open('dataset_books', 'wb') as fp:
        pickle.dump(data_source_a, fp)
    with open('dataset_kitchen', 'wb') as fp:
        pickle.dump(data_source_b, fp)
    with open('dataset_electronics', 'wb') as fp:
        pickle.dump(data_source_c, fp)

