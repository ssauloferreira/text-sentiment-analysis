import pickle

import numpy as np
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

from neural_networks import convL, mlp
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
n = 400
src = 'dvd'
tgt = 'books'
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
data_source, _, label_source, _ = train_test_split(dataset_source.docs, dataset_source.labels, test_size=0.0,
                                                   random_state=42)
data_source = to_process(data_source, '6', 3)

data_target, _, label_target, _ = train_test_split(dataset_target.docs, dataset_target.labels, test_size=0.0,
                                                   random_state=42)
data_target = to_process(data_target, '6', 3)

# ----------------------------------- clustering ---------------------------------------------
print("Clustering...")
vocabulary_source = get_vocabulary(data_source)
print('Vocabulary source:', len(vocabulary_source))
vocab_source, scores_source = get_senti_representation(vocabulary_source, True)
vocabulary_target = get_vocabulary(data_target)
print('Vocabulary target:', len(vocabulary_target))
vocab_target, scores_target = get_senti_representation(vocabulary_target, True)

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

# --------------------------------- feature selection ---------------------------------------
common = []

print("Number of features source:", len(vocab_source))
print("Number of features target:", len(vocab_target))

for feature in vocab_source:
    if feature in vocab_target:
        common.append(feature)

features_source = common
print("Number of common features: ", len(common))

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

# ALSENT (Average-Lexical-SentiWordNet-TFIDF):
# Average TF score = average of the term frequence in the documents it occurs
# sentiment score = pos_score - (neg_score * -1)
# ALSENT = Average TF score * sentiment score * IDF

# ------------------------------ calculating ALSENT -------------------------------------
print("ALSENT...")


def get_average_tfidf(feature, data):
    total = []
    count_idf = 0
    idf = None
    for text in data:
        count = 0
        for word in text:
            if word == feature:
                count += 1
        if count > 0:
            count_idf += 1
            total.append(count)
    try:
        idf = np.log(len(data) / count_idf)
    except:
        print(feature)
    return np.mean(total), idf


def feature_weight(cluster):
    weighted_cluster = []
    common_words = []

    for word in cluster:
        if word in features_source:
            common_words.append(word)

    for word in cluster:
        if len(common_words) == 0:
            weighted_cluster.append([word, 0])
        elif word not in common_words:
            aux_map = dict.fromkeys(common_words)
            for key in aux_map.keys():
                aux_map[key] = 0
            tf = 0

            for text in data_source:
                if word in text:
                    tf += 1
                    for key in aux_map.keys():
                        if key in text:
                            aux = aux_map[key]
                            aux += 1
                            aux_map[key] = aux
            weight = 0
            try:
                weight = max(aux_map.values()) / tf
            except:
                pass
            weighted_cluster.append([word, weight])
    return weighted_cluster


weighted_source = []
for cluster in wclusters_source:
    aux = feature_weight(cluster)
    aux.sort(key=lambda x: x[1], reverse=True)
    weighted_source.append(aux)

weighted_target = []
for cluster in wclusters_target:
    aux = feature_weight(cluster)
    aux.sort(key=lambda x: x[1], reverse=True)
    weighted_target.append(aux)

grouped_features = {}
for i in range(len(grouped_s)):
    x = grouped_s[i]
    y = grouped_t[i]

    for j in range(len(weighted_source[x])):
        if j < len(weighted_target[y]):
            aux = weighted_source[x][j][0] + '_' + weighted_target[y][j][0]
            grouped_features[weighted_source[x][j][0]] = aux
            grouped_features[weighted_target[y][j][0]] = aux

print('Length of grouped features: ', len(grouped_features) / 2)
'''
sentcluster_source = []
for i in range(len(wclusters_source)):
    aux = []
    for j in range(len(wclusters_source[i])):
        avg_tf, idf = get_average_tfidf(wclusters_source[i][j], data_source)
        sent = sclusters_source[i][j]
        sent_value = sent[0] - sent[1]
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
'''
# -------------------------------- grouping features ---------------------------------------------
'''
print("Linking features")
grouped_features = {}
# print(grouped_s)
for i in range(len(grouped_s)):

    # print(i, len(wclusters_source[grouped_s[i]]), len(wclusters_target[grouped_t[i]]))

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
'''
# --------------------------------- substitute in datasets ------------------------------------
print("substituting")
data_source_aux = data_source.copy()
for i in range(len(data_source_aux)):
    for j in range(len(data_source_aux[i])):
        if data_source_aux[i][j] in grouped_features:
            data_source_aux[i][j] = grouped_features[data_source_aux[i][j]]

data_target_aux = data_target.copy()
for i in range(len(data_target_aux)):
    for j in range(len(data_target_aux[i])):
        if data_target_aux[i][j] in grouped_features:
            data_target_aux[i][j] = grouped_features[data_target_aux[i][j]]

# print(data_source)
# print(data_target)

# --------------------------------- feature selection ---------------------------------------
features_linked = list(dict.fromkeys(grouped_features.values()))
features = features_linked + features_source

print("Feature selection 2...")
cv_source = CountVectorizer(max_df=0.95, min_df=2, vocabulary=features)
x_source = cv_source.fit_transform(to_string(data_source_aux))

chi_stats, p_vals = chi2(x_source, label_source)
chi_res = sorted(list(zip(cv_source.get_feature_names(), chi_stats)),
                 key=lambda x: x[1], reverse=True)[0:5000]

features = []
for chi in chi_res:
    features.append(chi[0])

num_words = len(features)

print('Final features: ', features)
# ------------------------------------ tf-idf -----------------------------------------------
cv = TfidfVectorizer(smooth_idf=True, norm='l1', vocabulary=features)
x_train = cv.fit_transform(to_string(data_source_aux))
x_test = cv.fit_transform(to_string(data_target_aux))
# tokenizer = Tokenizer(num_words=num_words)
# tokenizer.fit_on_texts(data_source_aux)
#
# vocab_size = len(tokenizer.word_index) + 1
#
# X_train = tokenizer.texts_to_sequences(data_source_aux)
# X_test = tokenizer.texts_to_sequences(data_target_aux)
#
# maxlen = 100
#
# x_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
# x_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
#
# print(x_train)
# print(x_test)

#  -------------------------------------- classifying  ---------------------------------------
print("classifying")

# mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
#                     beta_1=0.9, beta_2=0.999, early_stopping=False,
#                     epsilon=1e-08, hidden_layer_sizes=(5, 2),
#                     learning_rate='constant', learning_rate_init=0.001,
#                     max_iter=200, momentum=0.9, n_iter_no_change=10,
#                     nesterovs_momentum=True, power_t=0.5, random_state=1,
#                     shuffle=True, solver='lbfgs', tol=0.0001,
#                     validation_fraction=0.1, verbose=False, warm_start=False)
# print(label_target)
# print(label_source)
# mlp.fit(x_train, label_source)
# predict = mlp.predict(x_test)
#
# precision = f1_score(label_target, predict, average='binary')
# print('Precision:', precision)
# accuracy = accuracy_score(label_target, predict)
# print('Accuracy: ', accuracy)
# recall = recall_score(label_target, predict, average='binary')
# print('Recall: ', recall)
# confMatrix = confusion_matrix(label_target, predict)
# print('Confusion matrix: \n', confMatrix)
#
# classifier = LogisticRegression()
# classifier.fit(x_train, label_source)
# predict = classifier.predict(x_test)
#
# precision = f1_score(label_target, predict, average='binary')
# print('Precision:', precision)
# accuracy = accuracy_score(label_target, predict)
# print('Accuracy: ', accuracy)
# recall = recall_score(label_target, predict, average='binary')
# print('Recall: ', recall)
# confMatrix = confusion_matrix(label_target, predict)
# print('Confusion matrix: \n', confMatrix)
#
y_train = np_utils.to_categorical(label_target, 2)
y_test = np_utils.to_categorical(label_source, 2)

# model = convL(input_shape=vocab_size, embedding_size=50, input_len=maxlen)
model = mlp(input_shape=num_words)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

print(str(n), "clusters")

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
scores = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print("\n-------------------------------------------------------------\n")
