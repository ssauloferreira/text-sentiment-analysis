import nltk
import numpy as np
import spacy
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords, wordnet

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
    'JJ': 'JJ',
    'JJR': 'JJ',
    'JJS': 'JJ',
    'VB': 'VB',
    'VBD': 'VB',
    'VBG': 'VB',
    'VBN': 'VB',
    'VBP': 'VB',
    'VBZ': 'VB',
    'NN': 'NN',
    'NNS': 'NN',
    'NNP': 'NNP',
    'NNPS': 'NNP',
    'RB': 'RB',
    'RBR': 'RB',
    'RBS': 'RB'
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
                    result_pos.append([word[0], resume[word[1]]])
        else:
            result_pos = tokens_filtered

        new_docs.append(result_pos)

    return new_docs


def vocabulary_pos(dataset):
    vocab = []

    for text in dataset:
        for word in text:
            vocab.append([word[0], word[1]])

    return vocab


def get_senti_representation(vocabulary):
    vocab = []
    scores = []
    get_pos = {
        'NN': 'n',
        'VB': 'v',
        'JJ': 'a',
        'RB': 'r',
        'NNP': 'n'
    }

    for item in vocabulary:

        if len(item) != 2:
            raise Exception("not a [word,pos] list")

        word = item[0]
        pos = get_pos[item[1]]

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

            scores.append([np.mean(pos_score), np.mean(neg_score), np.mean(obj_score)])
        else:
            scores.append([0, 0, 0])
        vocab.append(word)

    return vocab, scores
