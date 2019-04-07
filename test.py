from gensim.models import KeyedVectors

src = 'games'

word = ''
path = 'Embeddings_Models/model_' + src + '.bin'
model = KeyedVectors.load_word2vec_format(path)
while word != 'exit':
    word = input('type a word: ')

    if word in model.vocab:
        print(model.most_similar(positive='bad'))
    else:
        print('this word is not in vocabulary. try other.')
