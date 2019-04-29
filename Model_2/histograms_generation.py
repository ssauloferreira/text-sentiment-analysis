import nltk
import spacy
from nltk import WordNetLemmatizer

from Model_2.pre_processing import to_process, get_wordnet_pos
import pickle

src = 'dvd'
lemmatizer = WordNetLemmatizer()
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

with open('Datasets/dataset_' + src, 'rb') as fp:
    dataset = pickle.load(fp)

_data = to_process(dataset.docs, '6', 5)

for p in (0, 1, 2):

    qtd = {
        'JJ': 0,
        'VB': 0,
        'NN': 0,
        'RB': 0,
        'NNP': 0
    }

    data = []

    if p == 2:
        data = _data
    else:
        for i in range(len(_data)):
            if dataset.labels[i] == p:
                data.append(_data[i])

    for text in data:
        for word in text:
            pos = resume[word[1]]
            aux = qtd[pos]
            aux += 1
            qtd[pos] = aux

    for item in qtd:
        print(item, ':', qtd[item])
