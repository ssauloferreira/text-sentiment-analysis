import string
import nltk
import numpy as np
from nltk import WordNetLemmatizer
from scipy.misc import comb
from stemming.porter2 import stem
from nltk.corpus import stopwords
from keras.preprocessing.text import text_to_word_sequence

lemmatizer = WordNetLemmatizer()
punctuation = "[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]:"

def to_process(docs, pos):

    # Reading stop-words
    arq = open('Preprocess/sw.txt', 'r')
    stop_words = set(stopwords.words('english'))

    new_docs = []

    for text in docs:

        # Tokenizing
        tokens = nltk.word_tokenize(text)

        result = []

        # Removing stop words
        for word in tokens:
            if word not in stopWords and word not in punctuation:
                result.append(word)

        # Stemming and lemmatizing
        stems = [stem(word) for word in tokens]
        result = [lemmatizer.lemmatize(word) for word in stems]

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

def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)
