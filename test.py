import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from classes import Dataset
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from pre_processing import to_process, training_word2vec
from sklearn.model_selection import ParameterGrid

model = training_word2vec()


def extend_features(features_source, text_data, num_of_words=10000):
    vocabulary = {}
    features_added = []

    for text in text_data:
        for word in text:
            if word not in vocabulary:
                vocabulary[word] = 0
            else:
                count = vocabulary[word]
                count = count + 1
                vocabulary[word] = count

    vocabulary = sorted(vocabulary.items(), key=lambda kv: kv[1])

    filtered_vocabulary = []

    i = 0
    for word in vocabulary:
        i = i + 1
        filtered_vocabulary.append(word)

        if i >= num_of_words:
            break

    for feature in features_source:
        for word in filtered_vocabulary:
            if model.similarity(feature, word) > 0.5:
                if word not in features_source:
                    features_added.append(word)
                    print(word)

    return features_added + features_source


def to_string(lists):
    new_docs = []

    for item in lists:
        text = ""
        for word in item:
            text = text + " " + word
        new_docs.append(text)

    return new_docs


# ---------------------------------------- parameters -------------------------------------------
params = {
    'num_of_features_source': [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000,
                               9500, 10000],
    'num_of_features_target': [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
    'feature_mode': ['union', 'intersec']}

num_of_features_source = 3000
num_of_features_target = 1000
pos = '6'

# --------------------------------------- loading datasets ---------------------------------------
with open('Datasets/dataset_games_10k', 'rb') as fp:
    source_a = pickle.load(fp)
with open('Datasets/dataset_music_10k', 'rb') as fp:
    source_b = pickle.load(fp)
with open('Datasets/dataset_kindle_10k', 'rb') as fp:
    target = pickle.load(fp)

# --------------------------------------- obs ---------------------------------------
print('Model 2 Cross-Domain.\n2 source domains and 1 target domain.')
print("Partial results\nCount mode: TFIDF\nStop words, POS filter, tokenizing, lemmatizing and stemming.")
print("POS\n1: Adjectives\n2: Adverbs\n3: Nouns\n4: Verbs\n5: Adjectives and adverbs\n6: Adjectives, adverbs and nouns")

# ------------------------------------ preprocessing  ----------------------------------------
print('Preprocessing')
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

for parameter in ParameterGrid(params):
    # -------------------------------------- chi source A -----------------------------------------
    print('Chi-source')
    cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
    x_source_a = cv.fit_transform(to_string(data_source_a.docs))

    chi_stats, p_vals = chi2(x_source_a, data_source_a.labels)
    chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats)),
                     key=lambda x: x[1], reverse=True)[0:parameter['num_of_features_source']]

    features_a = []
    for chi in chi_res:
        features_a.append(chi[0])

    # --------------------------------------- chi source B -----------------------------------------
    cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
    x_source_b = cv.fit_transform(to_string(data_source_b.docs))

    chi_stats, p_vals = chi2(x_source_b, data_source_b.labels)
    chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats
                              )), key=lambda x: x[1], reverse=True)[0:parameter['num_of_features_source']]

    features_b = []
    for chi in chi_res:
        features_b.append(chi[0])

    #  ------------------------------------- features selection  ----------------------------------
    print('Features selection')

    features = []

    if parameter['feature_mode'] == 'intersec':
        for feature in features_a:
            if feature in features_b:
                features.append(feature)
    else:
        features = [a for a in features_b]
        for feature in features_a:
            if feature not in features:
                features.append(feature)

    # features = extend_features(features, target_train)

    # --------------------------- chi target ----------------------------------
    cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, vocabulary=features)
    x_target = cv.fit_transform(to_string(target_train))

    chi_stats, p_vals = chi2(x_target, labels_train)
    chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats
                              )), key=lambda x: x[1], reverse=True)[0:parameter['num_of_features_target']]

    features_target = []
    for chi in chi_res:
        features_target.append(chi[0])

    #  ----------------------------------------- tf-idf  -----------------------------------------
    print('TF-IDF')

    cv = TfidfVectorizer(smooth_idf=True, min_df=3, norm='l1', vocabulary=features_target)
    x_train_tfidf = cv.fit_transform(to_string(target_train))  # tfidf de treino, y_train é o vetor de label
    x_test_tfidf = cv.fit_transform(to_string(target_test))  # tfidf de teste, y_test é o vetor de labels

    #  -------------------------------------- classifying  ---------------------------------------

    print('----------------------------------------------------------------------\n')
    print('First domain\'s features: ', features_a.__len__())
    print('Second domain\'s features: ', features_b.__len__())
    print('Number of features after', parameter['feature_mode'], ': ', features.__len__())
    print('Number of features after selection:', features_target.__len__())

    mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                        beta_1=0.9, beta_2=0.999, early_stopping=False,
                        epsilon=1e-08, hidden_layer_sizes=(5, 2),
                        learning_rate='constant', learning_rate_init=0.001,
                        max_iter=200, momentum=0.9, n_iter_no_change=10,
                        nesterovs_momentum=True, power_t=0.5, random_state=1,
                        shuffle=True, solver='lbfgs', tol=0.0001,
                        validation_fraction=0.1, verbose=False, warm_start=False)
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
