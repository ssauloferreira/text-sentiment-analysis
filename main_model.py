import pickle
from keras import Sequential
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, LSTM, Dense, Activation
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from classes import Dataset
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from pre_processing import to_process


def to_string(lists):
    new_docs = []

    for item in lists:
        text = ""
        for word in item:
            text = text + " " + word
        new_docs.append(text)

    return new_docs


# ---------------------------------------- parameters -------------------------------------------
classifier = 'logreg'
num_of_features_source = 3000
num_of_features_target = 1000
pos = '6'
features_mode = 'union'

# --------------------------------------- loading datasets ---------------------------------------
with open('Datasets/dataset_movies_10k', 'rb') as fp:
    source_a = Dataset(pickle.load(fp))
with open('Datasets/dataset_music_10k', 'rb') as fp:
    source_b = Dataset(pickle.load(fp))
with open('Datasets/dataset_tripadvisor_10k', 'rb') as fp:
    target = Dataset(pickle.load(fp))

# --------------------------------------- obs ---------------------------------------
print('Model 2 Cross-Domain.\n2 source domains and 1 target domain.')
print("Partial results\nCount mode: TFIDF\nStop words, POS filter, tokenizing, lemmatizing and stemming.")
print("POS\n1: Adjectives\n2: Adverbs\n3: Nouns\n4: Verbs\n5: Adjectives and adverbs\n6: Adjectives, adverbs and nouns")

# ------------------------------------ preprocessing  ----------------------------------------
target_train, target_test, labels_train, labels_test = train_test_split(target.docs, target.labels,
                                                                        train_size=0.8, random_state=42)

data_source_a = Dataset()
data_source_b = Dataset()

data_source_a.labels = source_a.labels
data_source_b.labels = source_b.labels

data_source_a.docs = to_process(source_a.docs, pos)
data_source_b.docs = to_process(source_b.docs, pos)
target_train = to_process(target_train, pos)
target_test = to_process(target_test, pos)

# -------------------------------------- chi source A -----------------------------------------
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

            # --------------------------- chi target ----------------------------------
cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, vocabulary=features)
x_target = cv.fit_transform(to_string(target_train))

chi_stats, p_vals = chi2(x_target, labels_train)
chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats
                          )), key=lambda x: x[1], reverse=True)[0:num_of_features_target]

features_target = []
for chi in chi_res:
    features_target.append(chi[0])

#  ----------------------------------------- tf-idf  -----------------------------------------

cv = TfidfVectorizer(smooth_idf=True, min_df=3, norm='l1', vocabulary=features_target)
x_train_tfidf = cv.fit_transform(to_string(target_train))  # tfidf de treino, y_train é o vetor de label
x_test_tfidf = cv.fit_transform(to_string(target_test))  # tfidf de teste, y_test é o vetor de labels

#  -------------------------------------- classifying  ---------------------------------------


print('----------------------------------------------------------------------\nClassifier: ', classifier,
      '\n')
print('First domain\'s features: ', features_a.__len__())
print('Second domain\'s features: ', features_b.__len__())
print('Number of features after', features_mode, ': ', features.__len__())
print('Number of features after selection:', features_target.__len__())

if classifier == 'cnn':
    model = Sequential()
    model.add(Embedding(500, 100, input_length=len(features_target)))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_tfidf, np.array(labels_train), validation_split=0.4, epochs=3, verbose=1)
    scores = model.evaluate(x_test_tfidf, np.array(labels_test), verbose=0)
    print("POS: ", pos)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("\n-------------------------------------------------------------\n")

elif classifier == 'rnn':
    model = Sequential()
    model.add(Embedding(len(features), 100, input_length=len(features_target)))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_tfidf, np.array(labels_train), validation_split=0.4, epochs=3, verbose=1)
    scores = model.evaluate(x_test_tfidf, np.array(labels_test), verbose=1)
    print("POS: ", pos)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("\n-------------------------------------------------------------\n")

elif classifier == 'mlp':
    mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                        beta_1=0.9, beta_2=0.999, early_stopping=False,
                        epsilon=1e-08, hidden_layer_sizes=(5, 2),
                        learning_rate='constant', learning_rate_init=0.001,
                        max_iter=200, momentum=0.9, n_iter_no_change=10,
                        nesterovs_momentum=True, power_t=0.5, random_state=1,
                        shuffle=True, solver='lbfgs', tol=0.0001,
                        validation_fraction=0.1, verbose=True, warm_start=False)
    mlp.fit(x_train_tfidf, labels_train)
    predict = mlp.predict(x_test_tfidf)

    precision = f1_score(labels_test, predict, average='binary')
    print('Precision:', precision)
    accuracy = accuracy_score(labels_test, predict)
    print('Accuracy: ', accuracy)
    recall = recall_score(labels_test, predict, average='binary')
    print('Recall: ', recall)

    confMatrix = confusion_matrix(labels_test, predict)
    print('Confusion matrix: \n', confMatrix)

elif classifier == 'knn':
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x_train_tfidf, labels_train)

    predict = [neigh.predict(test) for test in x_test_tfidf]
    print(len(predict))

    precision = f1_score(labels_test, predict, average='binary')
    print('Precision:', precision)
    accuracy = accuracy_score(labels_test, predict)
    print('Accuracy: ', accuracy)
    recall = recall_score(labels_test, predict, average='binary')
    print('Recall: ', recall)
    confMatrix = confusion_matrix(labels_test, predict)
    print('Confusion matrix: \n', confMatrix)

elif classifier == 'dense':
    model = Sequential()

    model.add(Dense(512, input_shape=(len(features_target),)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_tfidf, np.array(labels_train), validation_split=0.4, epochs=3, verbose=1)
    scores = model.evaluate(x_test_tfidf, np.array(labels_test), verbose=0)
    print("POS: ", pos)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("\n-------------------------------------------------------------\n")

elif classifier == 'logreg':
    classifier = LogisticRegression()
    classifier.fit(x_train_tfidf, labels_train)
    predict = classifier.predict(x_test_tfidf)

    precision = f1_score(labels_test, predict, average='binary')
    print('Precision:', precision)
    accuracy = accuracy_score(labels_test, predict)
    print('Accuracy: ', accuracy)
    recall = recall_score(labels_test, predict, average='binary')
    print('Recall: ', recall)
    confMatrix = confusion_matrix(labels_test, predict)
