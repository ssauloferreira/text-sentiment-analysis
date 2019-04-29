'''
from k_means import TFIDF_KMeans
word = ['a', 'b', 'c', 'd', 'e',
        'f', 'g', 'h', 'i', 'j',
        'k', 'l', 'm', 'n', 'o']
itens = [[1, 3, 2], [5, 1, 3], [8, 5, 6], [2, 3, 1], [4, 7, 2],
         [6, 2, 3], [5, 8, 3], [7, 6, 7], [8, 8, 8], [1, 1, 1],
         [2, 2, 2], [3, 1, 2], [4, 5, 9], [9, 9, 9], [2, 8, 3]]

print('len', len(itens))

kmeans = TFIDF_KMeans(n_clusters=3, vocabulary=word, n_it=1000, random_state=42)
kmeans.cluster(itens)

for i in kmeans.get_clusters():
    print(i)

for i in kmeans.get_word_clusters():
    print(i)

print(kmeans.get_centers())
'''
import pickle

from pre_processing import to_process, get_senti_representation

src = 'books'

with open('Datasets/dataset_' + src, 'rb') as fp:
    dataset = pickle.load(fp)

_data = to_process(dataset.docs[:50], '6', 5)

vocab, scores = get_senti_representation(_data)

for i in range(len(scores)):
    print(vocab[i], ':', scores[i])
