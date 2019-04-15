import random

from scipy.spatial import distance


class K_Means:
    def __init__(self, random_state=-1, n_clusters=8, n_it=50, dist='euclidean'):
        self.n_clusters = n_clusters
        self.n_it = n_it
        self.random_state = random_state
        self.dist = dist
        self.vectors = []
        self.size = 0
        self.old_centers = []
        self.current_centers = []
        self.clusters = []


    def distance(self, a, b):
        if self.dist == 'hamming':
            return distance.hamming(a, b)
        elif self.dist == 'euclidean':
            return distance.euclidean(a, b)
        elif self.dist == 'cosine':
            return distance.cosine(a, b)

    def initialize_centers(self):
        indexes = []
        while len(indexes < self.n_clusters):
            a = random.randint(0, self.size)
            if a not in indexes:
                indexes.append(a)

        for a in indexes:
            self.current_centers.append(self.vectors[a])

    def cluster(self, vectors):
        self.size = len(vectors)
        self.vectors = vectors

        self.initialize_centers()

        count = 0



