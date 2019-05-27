import pickle

import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

from pre_processing import to_process, get_vocabulary, get_senti_representation


def to_string(lists):
    new_docs = []

    for item in lists:
        text = ""
        for word in item:
            text = text + " " + word
        new_docs.append(text)

    return new_docs


nfeature = 8000
n = 100
src = 'books'
tgt = 'electronics'

with open('Datasets/dataset_' + src, 'rb') as fp:
    dataset_source = pickle.load(fp)

with open('Datasets/dataset_' + tgt, 'rb') as fp:
    dataset_target = pickle.load(fp)

# ---------------------------------- preprocessing -------------------------------------------
data_source = to_process(dataset_source.docs, '6', 0)
label_source = dataset_source.labels
data_target = to_process(dataset_target.docs, '6', 0)
label_target = dataset_target.labels

vocabulary_source = get_vocabulary(data_source)
vocab_source, scores_source = get_senti_representation(vocabulary_source, True)
vocabulary_target = get_vocabulary(data_target)
vocab_target, scores_target = get_senti_representation(vocabulary_target, True)

# ----------------------------------- clustering ---------------------------------------------
# clustering = DBSCAN(eps=1, min_samples=2)
# clustering = SpectralClustering(n_clusters=n, assign_labels="discretize", random_state=0)
clustering_source = KMeans(n_clusters=n, random_state=0)
clustering_source.fit(scores_source)
clustering_target = KMeans(n_clusters=n, random_state=0)
clustering_target.fit(scores_target)

wclusters_source = [[] for i in range(n)]
sclusters_source = [[] for i in range(n)]
wclusters_target = [[] for i in range(n)]
sclusters_target = [[] for i in range(n)]

for i in range(len(vocab_source)):
    aux = clustering_source.labels_[i]
    wclusters_source[aux].append(vocab_source[i])
    sclusters_source[aux].append(scores_source[i])

for i in range(len(vocab_target)):
    aux = clustering_target.labels_[i]
    wclusters_target[aux].append(vocab_target[i])
    sclusters_target[aux].append(scores_target[i])

# for text in data_source:
#    print(text)

# --------------------------------- feature selection ---------------------------------------
cv_source = CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
x_source = cv_source.fit_transform(to_string(data_source))

chi_stats, p_vals = chi2(x_source, label_source)
chi_res = sorted(list(zip(cv_source.get_feature_names(), chi_stats)),
                 key=lambda x: x[1], reverse=True)[0:nfeature]

features_source = []
for chi in chi_res:
    features_source.append(chi[0])

aux = []

print(len(features_source))
print(len(vocab_target))

for feature in features_source:
    if feature in vocab_target:
        aux.append(feature)

features_source = aux
# --------------------------------- agrupando clusters --------------------------------------
grouped_s = []
grouped_t = []

for i in range(n):

    aux_features = []
    for feature in features_source:
        if feature in wclusters_source[i]:
            aux_features.append(feature)

    index = 0
    sim = 0

    #print(aux_features)

    for j in range(n):
        if j not in grouped_t:
            count = 0
            for feature in aux_features:
                if feature in wclusters_target[j]:
                    count += 1

            if count > sim:
                sim = count
                index = j

    if index not in grouped_t:
        grouped_s.append(i)
        grouped_t.append(index)

        print(i, ' ', index, ': ', sim)
'''
avg_cluster_src = []
avg_cluster_tgt = []

for cluster in sclusters_source:
    pos = []
    neg = []
    obj = []
    for item in cluster:
        pos.append(item[0])
        neg.append(item[1])
        obj.append(item[2])
    result = [np.mean(pos), np.mean(neg), np.mean(obj)]
    avg_cluster_src.append(result)

for cluster in sclusters_target:
    pos = []
    neg = []
    obj = []
    for item in cluster:
        pos.append(item[0])
        neg.append(item[1])
        obj.append(item[2])
    result = [np.mean(pos), np.mean(neg), np.mean(obj)]
    avg_cluster_tgt.append(result)

for i in range(n):
    index = 0
    dst = np.inf

    for j in range(n):
        aux = distance.euclidean(avg_cluster_src[i], avg_cluster_tgt[j])
        if aux < dst:
            dst = aux
            index = j

    grouped_s.append(i)
    grouped_t.append(index)
'''
for n in range(len(grouped_s)):
    print('group ', n)
    print(wclusters_source[grouped_s[n]])
    print(wclusters_target[grouped_t[n]])

    print('\n\n')

#for i in range(len(grouped_s)):
#
# sentiment score = pos_score - (neg_score * -1)