import pickle

from gensim.models import KeyedVectors
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from sklearn.neural_network import MLPClassifier


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

def get_similar(common, src):
    path = 'Embeddings_Models/model_' + src[0] + '.bin'
    model1 = KeyedVectors.load_word2vec_format(path)
    path = 'Embeddings_Models/model_' + src[1] + '.bin'
    model2 = KeyedVectors.load_word2vec_format(path)

    new_com = []
    similar_a = []
    similar_b = []

    for word in common:
        _break = True
        w1 = ' '
        w2 = ' '
        if word in model1.vocab and word in model2:
            sim_a = model1.most_similar(positive=word)
            sim_b = model2.most_similar(positive=word)

            for sim in sim_a:
                if sim[0] not in common and sim[0] not in similar_a and sim[0] not in similar_b:
                    w1 = sim[0]

            for sim in sim_b:
                if sim[0] not in common and sim[0] not in similar_a and sim[0] not in similar_b and sim[0] != w1:
                    w2 = sim[0]

            if not w1.isspace() and not w2.isspace() and w1 != w2:
                new_com.append(word)
                similar_a.append(w1)
                similar_b.append(w2)

    return new_com, similar_a, similar_b

# ---------------------------------------- parameters -------------------------------------------
classifier = 'mlp'
num_of_features_source = 5000
pos = '6'
features_mode = 'intersec'
src = ['games', 'kindle']

# --------------------------------------- loading datasets ---------------------------------------
with open('Datasets/dataset_'+src[0], 'rb') as fp:
    data_source = pickle.load(fp)
with open('Datasets/dataset_'+src[1], 'rb') as fp:
    data_target = pickle.load(fp)

vocabulary_source = gen_vocab(data_source.docs)
vocabulary_target = gen_vocab(data_target.docs)

# -------------------------------------- chi source -----------------------------------------
print('Chi-source')
cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
x_source = cv.fit_transform(to_string(data_source.docs))

chi_stats, p_vals = chi2(x_source, data_source.labels)
chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats)),
                 key=lambda x: x[1], reverse=True)[0:num_of_features_source]

features_source = []
for chi in chi_res:
    features_source.append(chi[0])

# --------------------------------   feature selection  --------------------------------------

common = []

for item in features_source:
    if item in vocabulary_target:
        common.append(item)

new_common, exclusive_source, exclusive_target = get_similar(common, src)

print(new_common)
print(len(new_common))
print(exclusive_source)
print(len(exclusive_source))
print(exclusive_target)
print(len(exclusive_target))

features_source = new_common+exclusive_source
features_target = new_common+exclusive_target

#  ----------------------------------------- tf-idf  -----------------------------------------

cv = TfidfVectorizer(smooth_idf=True, min_df=3, norm='l1', vocabulary=features_source)
x_train_tfidf = cv.fit_transform(to_string(data_source.docs))
cv2 = TfidfVectorizer(smooth_idf=True, min_df=3, norm='l1', vocabulary=features_target)
x_test_tfidf = cv2.fit_transform(to_string(data_target.docs))

#  -------------------------------------- classifying  ---------------------------------------

if classifier == 'mlp':
    mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                        beta_1=0.9, beta_2=0.999, early_stopping=False,
                        epsilon=1e-08, hidden_layer_sizes=(5, 2),
                        learning_rate='constant', learning_rate_init=0.001,
                        max_iter=200, momentum=0.9, n_iter_no_change=10,
                        nesterovs_momentum=True, power_t=0.5, random_state=1,
                        shuffle=True, solver='lbfgs', tol=0.0001,
                        validation_fraction=0.1, verbose=False, warm_start=False)
    mlp.fit(x_train_tfidf, data_source.labels)
    predict = mlp.predict(x_test_tfidf)

    precision = f1_score(data_target.labels, predict, average='binary')
    print('Precision:', precision)
    accuracy = accuracy_score(data_target.labels, predict)
    print('Accuracy: ', accuracy)
    recall = recall_score(data_target.labels, predict, average='binary')
    print('Recall: ', recall)

    confMatrix = confusion_matrix(data_target.labels, predict)
    print('Confusion matrix: \n', confMatrix)

elif classifier == 'logreg':
    classifier = LogisticRegression()
    classifier.fit(x_train_tfidf, data_source.labels)
    predict = classifier.predict(x_test_tfidf)

    precision = f1_score(data_target.labels, predict, average='binary')
    print('Precision:', precision)
    accuracy = accuracy_score(data_target.labels, predict)
    print('Accuracy: ', accuracy)
    recall = recall_score(data_target.labels, predict, average='binary')
    print('Recall: ', recall)
    confMatrix = confusion_matrix(data_target.labels, predict)
    print('Confusion matrix: \n', confMatrix)

elif classifier == 'tree':
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train_tfidf, labels_train)
    predict = clf.predict(x_test_tfidf)

    precision = f1_score(data_target.labels, predict, average='binary')
    print('Precision:', precision)
    accuracy = accuracy_score(data_target.labels, predict)
    print('Accuracy: ', accuracy)
    recall = recall_score(data_target.labels, predict, average='binary')
    print('Recall: ', recall)
    confMatrix = confusion_matrix(data_target.labels, predict)
    print('Confusion matrix: \n', confMatrix)

print('----------------------------------------------------------------------\n')
