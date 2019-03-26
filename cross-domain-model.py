import numpy as np
import pickle

from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from extend_features import extend_features


def gen_vocab(dataset):
    vocabulary = []

    for text in dataset:
        for word in text:
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary


def to_string(lists):
    new_docs = []

    for item in lists:
        text = ""
        for word in item:
            text = text + " " + word
        new_docs.append(text)

    return new_docs


# ---------------------------------------- parameters -------------------------------------------
classifier = 'mlp'
num_of_features_source = 10000
num_of_features_target = 4000
pos = '6'
features_mode = 'intersec'
src = ['books', 'kitchen', 'electronics']

# --------------------------------------- loading datasets ---------------------------------------
with open('Datasets/dataset_'+src[0], 'rb') as fp:
    data_source_a = pickle.load(fp)
with open('Datasets/dataset_'+src[1], 'rb') as fp:
    data_source_b = pickle.load(fp)
with open('Datasets/dataset_'+src[2], 'rb') as fp:
    target = pickle.load(fp)

data = data_source_a.docs
labels = data_source_a.labels

# --------------------------------------- obs ---------------------------------------
print('Model 2 Cross-Domain.\n2 source domains and 1 target domain.')
print("Partial results\nCount mode: TFIDF\nStop words, POS filter, tokenizing, lemmatizing and stemming.")
print("POS\n1: Adjectives\n2: Adverbs\n3: Nouns\n4: Verbs\n5: Adjectives and adverbs\n6: Adjectives, adverbs and nouns")

# ----------------------------- preprocessing & splitting -------------------------------------

vocabulary = gen_vocab(target.docs)

# -------------------------------------- chi source A -----------------------------------------
print('Chi-source')
cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
x_source_a = cv.fit_transform(to_string(data_source_a.docs))

chi_stats, p_vals = chi2(x_source_a, data_source_a.labels)
chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats)),
                 key=lambda x: x[1], reverse=True)[0:num_of_features_source]

features_a = []
for chi in chi_res:
    features_a.append(chi[0])

# --------------------------------------- chi source B -----------------------------------------
cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
x_source_b = cv.fit_transform(to_string(data_source_b.docs))

chi_stats, p_vals = chi2(x_source_b, data_source_b.labels)
chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats
                          )), key=lambda x: x[1], reverse=True)[0:num_of_features_source]

features_b = []
for chi in chi_res:
    features_b.append(chi[0])

#  ------------------------------------- features selection  ----------------------------------
print('Features selection')

features = []

if features_mode == 'intersec':
    for feature in features_a:
        if feature in features_b:
            features.append(feature)
else:
    features = [a for a in features_b]
    for feature in features_a:
        if feature not in features:
            features.append(feature)

print('Features before expansion: ', len(features))

features = extend_features(features=features, vocabulary=vocabulary, src=src[2])


# --------------------------- chi target ----------------------------------
cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, vocabulary=features)
x_target = cv.fit_transform(to_string(target.docs))

chi_stats, p_vals = chi2(x_target, target.labels)
chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats
                          )), key=lambda x: x[1], reverse=True)[0:num_of_features_target]

features_target = []
for chi in chi_res:
    features_target.append(chi[0])
print(features_target)

#  ----------------------------------------- tf-idf  -----------------------------------------

cv = TfidfVectorizer(smooth_idf=True, min_df=3, norm='l1', vocabulary=features_target, max_features=num_of_features_target)
x_train_tfidf = cv.fit_transform(to_string(data))  # tfidf de treino, y_train é o vetor de label
x_test_tfidf = cv.fit_transform(to_string(target.docs))  # tfidf de teste, y_test é o vetor de labels

#  -------------------------------------- classifying  ---------------------------------------


print('First domain\'s features: ', features_a.__len__())
print('Second domain\'s features: ', features_b.__len__())
print('Number of features after expansion: ', features.__len__())

if classifier == 'mlp':
    mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                        beta_1=0.9, beta_2=0.999, early_stopping=False,
                        epsilon=1e-08, hidden_layer_sizes=(5, 2),
                        learning_rate='constant', learning_rate_init=0.001,
                        max_iter=200, momentum=0.9, n_iter_no_change=10,
                        nesterovs_momentum=True, power_t=0.5, random_state=1,
                        shuffle=True, solver='lbfgs', tol=0.0001,
                        validation_fraction=0.1, verbose=False, warm_start=False)
    mlp.fit(x_train_tfidf, labels)
    predict = mlp.predict(x_test_tfidf)

    precision = f1_score(target.labels, predict, average='binary')
    print('Precision:', precision)
    accuracy = accuracy_score(target.labels, predict)
    print('Accuracy: ', accuracy)
    recall = recall_score(target.labels, predict, average='binary')
    print('Recall: ', recall)

    confMatrix = confusion_matrix(target.labels, predict)
    print('Confusion matrix: \n', confMatrix)

elif classifier == 'knn':
    for k in range(10):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(x_train_tfidf, labels)

        predict = [neigh.predict(test) for test in x_test_tfidf]

        print('k:', k)

        precision = f1_score(target.labels, predict, average='binary')
        print('Precision:', precision)
        accuracy = accuracy_score(target.labels, predict)
        print('Accuracy: ', accuracy)
        recall = recall_score(target.labels, predict, average='binary')
        print('Recall: ', recall)
        confMatrix = confusion_matrix(target.labels, predict)
        print('Confusion matrix: \n', confMatrix)

elif classifier == 'logreg':
    classifier = LogisticRegression()
    classifier.fit(x_train_tfidf, labels)
    predict = classifier.predict(x_test_tfidf)

    precision = f1_score(target.labels, predict, average='binary')
    print('Precision:', precision)
    accuracy = accuracy_score(target.labels, predict)
    print('Accuracy: ', accuracy)
    recall = recall_score(target.labels, predict, average='binary')
    print('Recall: ', recall)
    confMatrix = confusion_matrix(target.labels, predict)

elif classifier == 'tree':
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train_tfidf, labels)
    predict = clf.predict(x_test_tfidf)

    precision = f1_score(target.labels, predict, average='binary')
    print('Precision:', precision)
    accuracy = accuracy_score(target.labels, predict)
    print('Accuracy: ', accuracy)
    recall = recall_score(target.labels, predict, average='binary')
    print('Recall: ', recall)
    confMatrix = confusion_matrix(target.labels, predict)

print('----------------------------------------------------------------------\n')
