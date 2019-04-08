from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, StackedEmbeddings

flair_forward = FlairEmbeddings('news-forward-fast')
flair_backward = FlairEmbeddings('news-backward-fast')

stacked_embeddings = StackedEmbeddings(embeddings=[
    flair_forward,
    flair_backward
])

sentence = Sentence('I like this kind of music')
stacked_embeddings.embed(sentence)

for token in sentence:
    print(token.embedding)
