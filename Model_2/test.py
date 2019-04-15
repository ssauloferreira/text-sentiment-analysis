from classes import Flair_Embedding


def gen_vocab(dataset):
    vocabulary = []

    for text in dataset:
        for word in text:
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary


'''
flair = Flair_Embedding()


print(0)
flair.gen_vocabulary(vocabulary)
print(1)
print(flair.most_similar('book'))
print(2)

'''

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