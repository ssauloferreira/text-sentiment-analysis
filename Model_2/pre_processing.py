import nltk
import numpy as np
import spacy
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords, wordnet

# python -m spacy download en

# Reading stop-words
stop_words = set(stopwords.words('english'))
# Loading spacy's model
nlp = spacy.load('en')
# Punctuation's list
punctuation = "[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]:0123456789 "
# Negation words
neg_words = ['not', 'no', 'nothing', 'never']
# Resume pos
resume = {
    'JJ': 'a',
    'JJR': 'a',
    'JJS': 'a',
    'VB': 'v',
    'VBD': 'v',
    'VBG': 'v',
    'VBN': 'v',
    'VBP': 'v',
    'VBZ': 'v',
    'NN': 'n',
    'NNS': 'n',
    'NNP': 'n',
    'NNPS': 'n',
    'RB': 'r',
    'RBR': 'r',
    'RBS': 'r'
}


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def tuple_to_list(tuples):
    result = []

    for tuple in tuples:
        object = [tuple[0], tuple[1]]
        result.append(object)

    return result


def rare_features(dataset, minimum_tf):
    vocabulary = {}

    for text in dataset:

        tokens = []
        doc = nlp(text)
        for word in doc:
            tokens.append(word.lemma_)

        for word in tokens:
            if word not in vocabulary:
                vocabulary[word] = 1
            else:
                aux = vocabulary[word]
                aux += 1
                vocabulary[word] = aux
    result = []

    for word in vocabulary:
        if vocabulary[word] < minimum_tf:
            result.append(word)

    return result


def negation_processing(text):
    negable = ['JJ', 'VB']
    size = len(text)

    text = tuple_to_list(text)

    for i in range(size):
        if text[i][0] in neg_words:
            j = i + 1
            while j < size:
                if text[j][1] in negable:
                    word_reverse = text[j][0] + '_' + 'not'
                    text[j][0] = word_reverse
                    break
                j += 1
    return text


def to_process(docs, pos, minimum_tf):
    # Loading rare rare_features
    rare = rare_features(docs, minimum_tf)

    new_docs = []

    for text in docs:

        # Tokenizing & lemmatization
        tokens = []
        doc = nlp(text)
        for word in doc:
            if word.lemma_ != '-PRON-':
                tokens.append(word.lemma_)

        # POS filter: only adverbs, adjectives and nouns
        pos_tags = nltk.pos_tag(tokens)
        pos_tags = negation_processing(pos_tags)

        # Removing stop words & punctuation & rare features
        tokens_filtered = []
        for word in pos_tags:
            if word[0] not in stop_words and word[0] not in punctuation and word[0] not in rare and word[0].isalpha():
                word[0] = word[0].lower()
                tokens_filtered.append(word)

        pos_tags = tokens_filtered
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
                if word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS' or \
                        word[1] == 'VB' or word[1] == 'VBD' or word[1] == 'VBG' or \
                        word[1] == 'VBN' or word[1] == 'VBP' or word[1] == 'VBZ' or \
                        word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'NNP' or word[1] == 'NNPS' or \
                        word[1] == 'RB' or word[1] == 'RBR' or word[1] == 'RBS':
                    aux = word[0] + '_' + resume[word[1]]
                    result_pos.append(aux)
        else:
            result_pos = tokens_filtered

        new_docs.append(result_pos)

    return new_docs


def get_vocabulary(dataset):
    vocab = {}

    for text in dataset:
        for word in text:
            if word not in vocab:
                vocab[word] = 0

    return vocab.keys()


def get_senti_representation(vocabulary, pos_form=False):
    vocab = []
    scores = []

    for item in vocabulary:

        word = ''
        pos = ''

        for i in range(len(item)):
            if item[i] == '_':
                word = item[:i]
                pos = item[i+1:]

        syns = list(swn.senti_synsets(word))
        if syns.__len__() > 0:
            pos_score = []
            neg_score = []
            obj_score = []
            for syn in syns:
                if pos in syn.synset.name():
                    pos_score.append(syn.pos_score())
                    neg_score.append(syn.neg_score())
                    obj_score.append(syn.obj_score())

            if len(pos_score) > 0:
                scores.append(
                    [round(max(pos_score), 3),
                     round(max(neg_score), 3),
                     round(max(obj_score), 3)]
                )
            else:
                scores.append([0, 0, 0])
        else:
            scores.append([0, 0, 0])

        if pos_form:
            vocab.append(word + '_' + pos)
        else:
            vocab.append(word)

    return vocab, scores
