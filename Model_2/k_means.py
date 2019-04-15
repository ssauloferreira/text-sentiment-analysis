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
        self.current_centers = self.initialize_clusters
        self.old_clusters = []
        self.current_clusters = []

    def average_cluster(self, cluster):
        for i in range(len(cluster[0])):
#####################################################################            

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

    def initialize_clusters(self):
        init = [0 for i in range(len(self.n_clusters))]
        return init

    def cluster(self, vectors):
        self.size = len(vectors)
        self.vectors = vectors

        self.initialize_centers()

        count = 0

        while count < self.m_it:

            self.old_centers = self.current_centers
            self.old_clusters = self.current_clusters
            self.current_clusters = self.initialize_clusters

            for vector in self.vectors:
                index = -1
                min = -1
                for i in len(self.old_centers):
                    dist = self.distance(vector, self.old_centers[i])
                    if dist < min:
                        index = i
                        i = min

                self.current_clusters[index].append(vector)

            avg_cluster = []

            for cluster in current_clusters:
                avg_cluster = self.average_cluster(cluster)
