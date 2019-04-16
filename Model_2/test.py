'''
from classes import Flair_Embedding


def gen_vocab(dataset):
    vocabulary = []

    for text in dataset:
        for word in text:
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary



flair = Flair_Embedding()


print(0)
flair.gen_vocabulary(vocabulary)
print(1)
print(flair.most_similar('book'))
print(2)



flair = Flair_Embedding()


with open('Datasets/dataset_books', 'rb') as fp:
    data_source = pickle.load(fp)

docs = data_source.docs

docs = docs[:][0]

print(docs[0], docs[1])

vocabulary = gen_vocab(docs)

qtd = 0
size = len(vocabulary)

words = []

print('Quantidade total:', size)

for word in vocabulary:
    try:
        flair.embed_unique(word)
    except:
        qtd += 1
        words.append(word)

print('Quantidade faltante:', qtd)
print('Palavras faltantes')
print(words)

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
