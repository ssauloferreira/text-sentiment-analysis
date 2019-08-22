import random

from numpy import mean, math
from scipy.spatial import distance


class TFIDF_KMeans:
    def __init__(self, vocabulary, random_state=-1, n_clusters=8, n_it=50, dist='euclidean'):
        self.n_clusters = n_clusters
        self.n_it = n_it
        self.random_state = random_state
        self.dist = dist
        self.vectors = []
        self.size = 0
        self.old_centers = []
        self.current_centers = self.initialize_clusters
        self.old_clusters = []
        self.current_clusters = []
        self.clusterized = False

        self.vocabulary = vocabulary
        self.current_clusters_vocab = self.initialize_clusters()
        self.old_clusters_vocab = []

    def get_clusters(self):
        if self.clusterized:
            return self.current_clusters
        raise Exception("Not clusterized")

    def get_word_clusters(self):
        if self.clusterized:
            return self.current_clusters_vocab
        raise Exception("Not clusterized")

    def get_centers(self):
        if self.clusterized:
            return self.current_centers
        raise Exception("Not clusterized")

    def check_difference(self):
        for i in self.current_centers:
            for j in self.old_centers:
                if i != j:
                    return True
        return False

    def average_cluster(self, cluster):
        avg = []
        for i in range(len(cluster[0])):
            a = [vec[i] for vec in cluster]
            avg.append(mean(a))
        return avg

    def distance(self, a, b):
        if self.dist == 'hamming':
            return distance.hamming(a, b)
        elif self.dist == 'euclidean':
            return distance.euclidean(a, b)
        elif self.dist == 'cosine':
            return distance.cosine(a, b)

    def initialize_centers(self):
        indexes = []

        if self.random_state > -1:
            random.seed(self.random_state)

        while len(indexes) < self.n_clusters:
            a = random.randint(0, self.size - 1)
            if a not in indexes:
                indexes.append(a)
        print(indexes)
        aux = []
        for a in indexes:
            print(a)
            aux.append(self.vectors[a])

        self.current_centers = aux

    def initialize_clusters(self):
        init = []
        for i in range(self.n_clusters):
            a = []
            init.append(a)
        return init

    def cluster(self, vectors):
        self.size = len(vectors)
        self.vectors = vectors

        self.initialize_centers()

        count = 0
        aux = []

        while count < self.n_it:

            if len(aux) > 0:
                self.current_centers = aux

            self.old_centers = self.current_centers

            self.old_clusters = self.current_clusters
            self.old_clusters_vocab = self.current_clusters_vocab

            self.current_clusters = self.initialize_clusters
            self.current_clusters_vocab = self.initialize_clusters()

            aux = self.initialize_clusters()
            aux_vocab = self.initialize_clusters()

            j = 0
            for vector in self.vectors:
                index = -1
                min = math.inf
                for i in range(len(self.old_centers)):
                    dist = self.distance(vector, self.old_centers[i])
                    if dist < min:
                        index = i
                        min = dist

                aux[index].append(vector)
                aux_vocab[index].append(self.vocabulary[j])

                j += 1

            self.current_clusters = aux
            self.current_clusters_vocab = aux_vocab

            aux = []
            j = 0
            for cluster in self.current_clusters:
                avg_cluster = self.average_cluster(cluster)
                aux.append(avg_cluster)
                j += 1

            if not self.check_difference():
                break

            count += 1

        self.clusterized = True
