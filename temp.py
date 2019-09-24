import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from math import pow
from sklearn.feature_extraction.text import TfidfVectorizer
from pre_processing import to_process, get_senti_representation, get_vocabulary

def to_string(lists):
    new_docs = []

    for item in lists:
        text = ""
        for word in item:
            text = text + " " + word
        new_docs.append(text)

    return new_docs

with open('dictionary', 'rb') as fp:
    grouped = pickle.load(fp)

with open('Datasets/dataset_kitchen', 'rb') as fp:
    dataset = pickle.load(fp)
data_src = to_process(dataset.docs, '1011', 3)

with open('Datasets/dataset_electronics', 'rb') as fp:
    dataset = pickle.load(fp)
data_tgt = to_process(dataset.docs, '1011', 3)

vocabulary_source = get_vocabulary(data_src)
vocab_source, scores_source, dicti_source = get_senti_representation(vocabulary_source, True)

vocabulary_target = get_vocabulary(data_tgt)
vocab_target, scores_target, dicti_target = get_senti_representation(vocabulary_target, True)

x = []
y = []
s = []
color = []

cv = TfidfVectorizer(smooth_idf=True, norm='l1', vocabulary=vocab_source)
source = cv.fit_transform(to_string(data_src))

aux1 = np.zeros((source.shape[1], source.shape[0]))

for i in range(source.shape[0]):
    for j in range(source.shape[1]):
        aux1[j][i] = source.tocsr()[i,j]

cv = TfidfVectorizer(smooth_idf=True, norm='l1', vocabulary=vocab_target)
target = cv.fit_transform(to_string(data_tgt))

print(1)
aux2 = np.zeros((target.shape[1], target.shape[0]))
for i in range(target.shape[0]):
    for j in range(target.shape[1]):
        aux2[j][i] = target.tocsr()[i,j]

pca = PCA(n_components=2)
pca_src = pca.fit_transform(aux1)
pca_tgt = pca.fit_transform(aux2)

i = 0
for item in dicti_source:
    value = dicti_source[item]

    a = pca_src[i][0] * 10
    b = pca_src[i][1] * 10

    if item in dicti_target:
        pass
    elif item in grouped:
        color.append('k')
        x.append(a/10)
        y.append(b/10)
        s.append(5)
    else:
        color.append('c')
        x.append(a)
        y.append(b)
        s.append(5)
    i += 1

i = 0
for item in dicti_target:
    value = dicti_target[item]

    if item in dicti_source:
        a = pca_tgt[i][0]
        b = pca_tgt[i][1]
        color.append('k')
        x.append(a)
        y.append(b)
        s.append(5)
    elif item in grouped:
        a = pca_tgt[i][0]
        b = pca_tgt[i][1]
        color.append('k')
        x.append(a)
        y.append(b)
        s.append(5)
    else:
        a = pca_tgt[i][0] * -10
        b = pca_tgt[i][1] * 10
        x.append(a)
        y.append(b)
        s.append(5)
        color.append('c')
    i += 1

plt.scatter(x, y, s, color=color)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.savefig('model.png')
plt.show()

'''
for item in dicti_source:
    value = dicti_source[item]
    print(value)
    x.append(value[0])
    y.append(value[1])
    s.append(10)
    color.append('b') if value[0] > value[1] else color.append('r')

plt.scatter(x, y, s, color=color)
plt.xlim([0,1])
plt.ylim([0,1])
plt.savefig('kitchen.png')
plt.show()
'''