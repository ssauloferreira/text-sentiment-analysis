import pickle

import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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
n = 300
src = 'books'
tgt = 'electronics'
maxlen = 500
batch_size = 32
filters = 250
kernel_size = 5
hidden_dims = 250
epochs = 3
nb_epoch_t = 50
_ = None

with open('Datasets/dataset_' + src, 'rb') as fp:
    dataset_source = pickle.load(fp)

with open('Datasets/dataset_' + tgt, 'rb') as fp:
    dataset_target = pickle.load(fp)

# ---------------------------------- preprocessing -------------------------------------------
data_source, _, label_source, _ = train_test_split(dataset_source.docs, dataset_source.labels, test_size=0.0, random_state=42)
data_source = to_process(data_source, '6', 0)
#label_source = dataset_source.labels

data_target, _, label_target, _ = train_test_split(dataset_target.docs, dataset_target.labels, test_size=0.0, random_state=42)
data_target = to_process(data_target, '6', 0)
#label_target = dataset_target.labels

vocabulary_source = get_vocabulary(data_source)
vocab_source, scores_source = get_senti_representation(vocabulary_source, True)
vocabulary_target = get_vocabulary(data_target)
vocab_target, scores_target = get_senti_representation(vocabulary_target, True)

# ----------------------------------- clustering ---------------------------------------------
print("Clustering...")
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
print("Feature selection 1...")
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
print("Linking clusters")
grouped_s = []
grouped_t = []

for i in range(n):

    aux_features = []
    for feature in features_source:
        if feature in wclusters_source[i]:
            aux_features.append(feature)

    index = 0
    sim = 0

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

# ALSENT (Average-Lexical-SentiWordNet-TFIDF):
# Average TF score = average of the term frequence in the documents it occurs
# sentiment score = pos_score - (neg_score * -1)
# ALSENT = Average TF score * sentiment score * IDF

# ------------------------------ calculating ALSENT -------------------------------------
print("ALSENT...")


def get_average_tfidf(feature, data_source):
    total = []
    count_idf = 0
    for text in data_source:
        count = 0
        for word in text:
            if word == feature:
                count += 1
        if count > 0:
            count_idf += 1
            total.append(count)
    return np.mean(total), np.log(len(data_source) / count_idf)


sentcluster_source = []
for i in range(len(wclusters_source)):
    aux = []
    for j in range(len(wclusters_source[i])):
        avg_tf, idf = get_average_tfidf(wclusters_source[i][j], data_source)
        sent = sclusters_source[i][j]
        sent_value = sent[0] + (-1 * sent[1])
        alsent = avg_tf * idf

        aux.append(alsent)

    sentcluster_source.append(aux)

sentcluster_target = []
for i in range(len(wclusters_target)):
    aux = []
    for j in range(len(wclusters_target[i])):
        avg_tf, idf = get_average_tfidf(wclusters_target[i][j], data_target)
        sent = sclusters_target[i][j]
        sent_value = sent[0] + (-1 * sent[1])
        alsent = avg_tf * sent_value * idf

        aux.append(alsent)

    sentcluster_target.append(aux)
print(len(sentcluster_source))
print(len(sentcluster_target))
# -------------------------------- grouping features ---------------------------------------------
print("Linking features")
grouped_features = {}
# print(grouped_s)
for i in range(len(grouped_s)):

    print(i, len(wclusters_source[grouped_s[i]]), len(wclusters_target[grouped_t[i]]))

    for j in range(len(wclusters_source[grouped_s[i]])):
        # print(len())
        if wclusters_source[grouped_s[i]][j] not in features_source:
            index = -1
            min = np.inf

            for k in range(len(wclusters_target[grouped_t[i]])):
                # print(i, j, k)
                if wclusters_target[grouped_t[i]][k] not in features_source and \
                        wclusters_target[grouped_t[i]][k] not in grouped_features:
                    dist = abs(sentcluster_source[grouped_s[i]][j] - sentcluster_target[grouped_t[i]][k])

                    if dist < min:
                        index = k
                        min = dist

            if index > -1:
                grouped_features[wclusters_source[grouped_s[i]][j]] = wclusters_source[grouped_s[i]][j] + "_" + \
                                                                      wclusters_target[grouped_t[i]][index]
                grouped_features[wclusters_target[grouped_t[i]][index]] = wclusters_source[grouped_s[i]][j] + "_" + \
                                                                          wclusters_target[grouped_t[i]][index]

print(grouped_features)

# --------------------------------- substitute in datasets ------------------------------------
print("substituting")
for i in range(len(data_source)):
    for j in range(len(data_source[i])):
        if data_source[i][j] in grouped_features:
            data_source[i][j] = grouped_features[data_source[i][j]]

for i in range(len(data_target)):
    for j in range(len(data_target[i])):
        if data_target[i][j] in grouped_features:
            data_target[i][j] = grouped_features[data_target[i][j]]

#print(data_source)
#print(data_target)

# --------------------------------- feature selection ---------------------------------------
aux = list(dict.fromkeys(grouped_features.values()))
vocabulary_aux = features_source

for item in aux:
    vocabulary_aux.append(item)

print(len(vocabulary_aux))

print("Feature selection 2...")
cv_source = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, vocabulary=vocabulary_aux)
x_source = cv_source.fit_transform(to_string(data_source))

chi_stats, p_vals = chi2(x_source, label_source)
chi_res = sorted(list(zip(cv_source.get_feature_names(), chi_stats)),
                 key=lambda x: x[1], reverse=True)[0:10000]

features = []
for chi in chi_res:
    features.append(chi[0])

'''
aux = []
for feature in features:
    if feature in vocab_target:
        aux.append(feature)

features = aux
'''
print('number of features = ', len(features))

print(features)
# ------------------------------------ tf-idf -----------------------------------------------
cv = TfidfVectorizer(smooth_idf=True, norm='l1', vocabulary=vocabulary_aux)
x_train = cv.fit_transform(to_string(data_source))
x_test = cv.fit_transform(to_string(data_target))

#  -------------------------------------- classifying  ---------------------------------------
print("classifying")
'''
mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                    beta_1=0.9, beta_2=0.999, early_stopping=False,
                    epsilon=1e-08, hidden_layer_sizes=(5, 2),
                    learning_rate='constant', learning_rate_init=0.001,
                    max_iter=200, momentum=0.9, n_iter_no_change=10,
                    nesterovs_momentum=True, power_t=0.5, random_state=1,
                    shuffle=True, solver='lbfgs', tol=0.0001,
                    validation_fraction=0.1, verbose=False, warm_start=False)
print(label_target)
print(label_source)
mlp.fit(x_train, label_source)
predict = mlp.predict(x_test)

precision = f1_score(label_target, predict, average='binary')
print('Precision:', precision)
accuracy = accuracy_score(label_target, predict)
print('Accuracy: ', accuracy)
recall = recall_score(label_target, predict, average='binary')
print('Recall: ', recall)
confMatrix = confusion_matrix(label_target, predict)
print('Confusion matrix: \n', confMatrix)

classifier = LogisticRegression()
classifier.fit(x_train, label_source)
predict = classifier.predict(x_test)

precision = f1_score(label_target, predict, average='binary')
print('Precision:', precision)
accuracy = accuracy_score(label_target, predict)
print('Accuracy: ', accuracy)
recall = recall_score(label_target, predict, average='binary')
print('Recall: ', recall)
confMatrix = confusion_matrix(label_target, predict)
print('Confusion matrix: \n', confMatrix)
'''
y_train = np_utils.to_categorical(label_target, 2)
y_test = np_utils.to_categorical(label_source, 2)

model = Sequential()
model.add(Dense(1000, input_shape=(len(vocabulary_aux),)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

print(model.summary())

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1)
model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)
