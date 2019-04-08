from scipy.spatial import distance


class Dataset:
    def __init__(self):
        self.docs = []
        self.labels = []

    def add(self, data):
        self.docs.append(data.doc)
        self.labels.append(data.label)

    @staticmethod
    def to_string(data):
        new_docs = []

        for doc in data:
            text = ""
            for word in doc:
                text = text + " " + word
            new_docs.append(text)

        return new_docs


class Data:
    def __init__(self, doc, label):
        self.doc = doc
        self.label = label

    def setlabel(self, newlabel):
        self.label = newlabel

    def tostring(self):
        return self.label, self.doc


class Flair_Embedding:
    def __init__(self):
        self.vocabulary = {}
        flair_forward = FlairEmbeddings('news-forward-fast')
        flair_backward = FlairEmbeddings('news-backward-fast')

        self.stacked_embeddings = StackedEmbeddings(embeddings=[
            flair_forward,
            flair_backward
        ])

    def gen_vocabulary(self, vocabulary):
        for word in vocabulary:
            self.vocabulary[word] = self.stacked_embeddings.embed(word)

    def embed_unique(self, word):
        return self.stacked_embeddings.embed(word)

    def similarity(self, word_a, word_b):
        if self.vocabulary.__len__() == 0:
            raise Exception('vocabulary is empty.')
        else:
            similarity = 1 / distance.euclidean(self.embedding(word_a), self.embedding(word_b))

        return similarity

    def most_similar(self, word, dictionary, num_of_similar):
        most_similar = []
        for word_b in dictionary:
            most_similar.append([word_b, self.similarity(word, word_b)])

        most_similar = sorted(most_similar, key=lambda x: x[1], reverse=True)

        return most_similar[:num_of_similar]

    def embedding(self, word):
        if word not in self.vocabulary:
            raise Exception('word is not in vocabulary.')
        else:
            return self.vocabulary[word]