import nltk
import spacy
import xlsxwriter
from nltk import WordNetLemmatizer

from pre_processing import to_process, get_wordnet_pos
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

book = xlsxwriter.Workbook('Sheets/histograms.xls')
for src in ('books', 'dvd', 'electronics', 'kitchen'):
    j = 0

    sheet = book.add_worksheet(src)

    with open('Datasets/dataset_' + src, 'rb') as fp:
        dataset = pickle.load(fp)

    _data = to_process(dataset.docs, '6', 5)

    for p in (0, 1, 2):

        i = 0

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
            for k in range(len(_data)):
                if dataset.labels[k] == p:
                    data.append(_data[k])

        for text in data:
            for word in text:
                pos = resume[word[1]]
                aux = qtd[pos]
                aux += 1
                qtd[pos] = aux
        if p == 0:
            sheet.write(i, j, 'negatives')
        elif p == 1:
            sheet.write(i, j, 'positives')
        else:
            sheet.write(i, j, 'all')

        i += 1

        for item in qtd:
            sheet.write(i, j, item)
            sheet.write(i, j+1, qtd[item])
            i += 1

        j += 3

book.close()