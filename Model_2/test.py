import pickle
import hdbscan
import xlsxwriter
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

from pre_processing import to_process, get_senti_representation, vocabulary_pos

src = 'books'
n = 10

for src in ('books', 'electronics', 'kitchen', 'dvd'):

    book = xlsxwriter.Workbook('Sheets/Clustering/DBScan' + src + '.xls')
    # book = xlsxwriter.Workbook(src + '.xls')

    with open('Datasets/dataset_' + src, 'rb') as fp:
        dataset = pickle.load(fp)

    # Preprocessing and getting swn representation
    data = to_process(dataset.docs, '6', 50)
    vocabulary = vocabulary_pos(data)
    vocab, scores = get_senti_representation(vocabulary, True)

    for n in (10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 350, 400):

        print(src, n, end=' ')

        clustering = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True, gen_min_span_tree=False,
        leaf_size=40, metric='euclidean',min_cluster_size=5, min_samples=None, p=None)

        # clustering = DBSCAN(eps=1, min_samples=2, algorithm='kd_tree')
        # clustering = SpectralClustering(n_clusters=n, assign_labels="discretize", random_state=0)
        # clustering = KMeans(n_clusters=n, random_state=0)

        clustering.fit(scores)

        clusters = [[] for i in range(max(clustering.labels_)+1)]

        print(len(vocab))
        print(len(clustering.labels_))

        for i in range(len(vocab)):
            if vocab[i] not in clusters[clustering.labels_[i]]:
                clusters[clustering.labels_[i]].append(vocab[i])

        sheet = book.add_worksheet(n.__str__() + ' clusters')

        for i in range(len(clusters)):
            for j in range(len(clusters[i])):
                sheet.write(j, i, clusters[i][j])

        break
    print()
    book.close()
