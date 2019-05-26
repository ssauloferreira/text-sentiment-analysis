import pickle

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

from pre_processing import to_process, vocabulary_pos, get_senti_representation

n = 100
src = 'books'
tgt = 'electronics'

with open('Datasets/dataset_' + src, 'rb') as fp:
    dataset_source = pickle.load(fp)

with open('Datasets/dataset_' + tgt, 'rb') as fp:
    dataset_target = pickle.load(fp)

# ---------------------------------- Preprocessing -------------------------------------------
data_source = to_process(dataset_source.docs, '6', 50)
label_source = dataset_source.labels
data_target = to_process(dataset_target.docs, '6', 50)
label_target = dataset_target.labels

vocabulary_source = vocabulary_pos(data_source)
vocab_source, scores_source = get_senti_representation(vocabulary_source, True)
vocabulary_target = vocabulary_pos(data_target)
vocab_target, scores_target = get_senti_representation(vocabulary_target, True)

# ----------------------------------- clustering ---------------------------------------------
# clustering = DBSCAN(eps=1, min_samples=2)
# clustering = SpectralClustering(n_clusters=n, assign_labels="discretize", random_state=0)
clustering_source = KMeans(n_clusters=n, random_state=0)
clustering_source.fit(scores_source)
clustering_target = KMeans(n_clusters=n, random_state=0)
clustering_target.fit(scores_source)

wclusters = [[] for i in range(n)]
sclusters = [[] for i in range(n)]

for i in range(len(vocab)):
    aux = clustering.labels_[i]
    wclusters[aux].append(vocab[i])
    sclusters[aux].append(scores[i])

cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
x_source = cv.fit_transform(to_string(data_source.docs))

chi_stats, p_vals = chi2(x_source, data_source.labels)
chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats)),
                 key=lambda x: x[1], reverse=True)[0:num_of_features_source]

features_source = []
for chi in chi_res:
    features_source.append(chi[0])