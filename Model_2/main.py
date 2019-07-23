import pickle

import gensim
import numpy as np
from gensim.models import Word2Vec
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.tree import tree

import neural_networks
from pre_processing import to_process, get_vocabulary, get_senti_representation


def to_string(lists):
    new_docs = []

    for item in lists:
        text = ""
        for word in item:
            text = text + " " + word
        new_docs.append(text)

    return new_docs


model = gensim.models.KeyedVectors.load_word2vec_format('Datasets/google-pretrained.bin', binary=True)
classif = "mlp"
vocabulary_size = 10000
embedding_size = 300
text_rep = 'embeddings'
pos = '1111'
num_layers = 200
nfeature = 8000
n = 2
src = 'kitchen'
tgt = 'electronics'
maxlen = 500
batch_size = 32
filters = 250
kernel_size = 5
hidden_dims = 250
epochs = 4
nb_epoch_t = 50
_ = None

get_bin = lambda x, n: format(x, 'b').zfill(n)

for src in ['books', 'dvd', 'electronics', 'kitchen']:
     for tgt in ['books', 'dvd', 'electronics', 'kitchen']:
         if src != tgt:
            with open('Datasets/dataset_' + src, 'rb') as fp:
                dataset_source = pickle.load(fp)

            with open('Datasets/dataset_' + tgt, 'rb') as fp:
                dataset_target = pickle.load(fp)

            # ---------------------------------- preprocessing -------------------------------------------
            print("\npreprocessing=====================================\n")
            data_source, _, label_source, _ = train_test_split(dataset_source.docs, dataset_source.labels,
                                                               test_size=0.01,
                                                               random_state=42)
            data_source = to_process(data_source, pos, 3)

            data_target, _, label_target, _ = train_test_split(dataset_target.docs, dataset_target.labels,
                                                               test_size=0.01,
                                                               random_state=42)
            data_target = to_process(data_target, pos, 3)

            # ----------------------------------- clustering ---------------------------------------------
            print("\nclustering=======================================\n")
            vocabulary_source = get_vocabulary(data_source)
            print('Vocabulary source:', len(vocabulary_source))
            vocab_source, scores_source, dicti_source = get_senti_representation(vocabulary_source, True)
            vocabulary_target = get_vocabulary(data_target)
            print('Vocabulary target:', len(vocabulary_target))
            vocab_target, scores_target, dicti_target = get_senti_representation(vocabulary_target, True)

            dicti = {}
            dicti.update(dicti_source)
            dicti.update(dicti_target)

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

            print("Number of sentiment features source:", len(vocab_source))
            print("Number of sentiment features target:", len(vocab_target))

            for feature in vocab_source:
                if feature in vocab_target:
                    common.append(feature)

            features_source = common
            print("Number of common features: ", len(common))

            # --------------------------------- agrupando clusters --------------------------------------
            print("\nlinking clusters========================================\n")
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
            print("\nconnecting features===================================\n")


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
                        aux = weighted_source[x][j][0][:-2] + '-' + weighted_target[y][j][0][:-2]
                        grouped_features[weighted_source[x][j][0]] = aux
                        grouped_features[weighted_target[y][j][0]] = aux

            print('Length of grouped features: ', len(grouped_features) / 2)

            # ----------------------------------- feature replacement ----------------------------------------
            print("\nsubstituting ==============================================\n")
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

            print("\nclassifying==============================================\n")
            if text_rep == 'tf-idf':
                # --------------------------------- feature selection ---------------------------------------
                features_linked = list(dict.fromkeys(grouped_features.values()))
                features = features_linked + features_source

                cv_source = CountVectorizer(max_df=0.95, min_df=2, vocabulary=features)
                x_source = cv_source.fit_transform(to_string(data_source_aux))

                chi_stats, p_vals = chi2(x_source, label_source)
                chi_res = sorted(list(zip(cv_source.get_feature_names(), chi_stats)),
                                 key=lambda x: x[1], reverse=True)[0:1500]

                features = []
                for chi in chi_res:
                    features.append(chi[0])

                num_words = len(features)

                # ------------------------------------ tf-idf -----------------------------------------------
                cv = TfidfVectorizer(smooth_idf=True, norm='l1', vocabulary=features)
                x_train = cv.fit_transform(
                    to_string(data_source_aux))  # alsent(dataset=data_source_aux, dicti=dicti, features=features)
                x_test = cv.transform(
                    to_string(data_target_aux))  # alsent(dataset=data_target_aux, dicti=dicti, features=features)

                # print(len(x_train), len(x_train[0]))
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
                if classif == "logistic regression":
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
                    print('\n')

                if classif == "deicion tree":
                    clf = tree.DecisionTreeClassifier()
                    clf.fit(x_train, label_source)
                    predict = clf.predict(x_test)

                    precision = f1_score(label_target, predict, average='binary')
                    print('Precision:', precision)
                    accuracy = accuracy_score(label_target, predict)
                    print('Accuracy: ', accuracy)
                    recall = recall_score(label_target, predict, average='binary')
                    print('Recall: ', recall)
                    confMatrix = confusion_matrix(label_target, predict)
                    print('Confusion matrix: \n', confMatrix)
                    print('\n')

                if classif == "mlp":
                    y_train = np_utils.to_categorical(label_target, 2)
                    y_test = np_utils.to_categorical(label_source, 2)

                    model = neural_networks.mlp(input_shape=num_words, num_layers=num_layers)
                    # model.summary()
                    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
                    # #
                    # # print(str(n), "clusters")
                    # #
                    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
                    scores = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
                    print("src", src, "tgt", tgt, "num_layers", num_layers, 'pos', pos)
                    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

            elif text_rep == 'embeddings':

                for text in data_source_aux:
                    for a in range(len(text)):
                        if '_' in text[a]:
                            text[a] = text[a][:-2]

                for text in data_target_aux:
                    for a in range(len(text)):
                        if '_' in text[a]:
                            text[a] = text[a][:-2]

                tokenizer = Tokenizer(num_words=vocabulary_size)
                tokenizer.fit_on_texts(data_source_aux + data_target_aux)

                sequences_src = tokenizer.texts_to_sequences(data_source_aux)
                sequences_tgt = tokenizer.texts_to_sequences(data_target_aux)

                maxlen = 100

                x_train = pad_sequences(sequences_src, padding='post', maxlen=maxlen)
                x_test = pad_sequences(sequences_tgt, padding='post', maxlen=maxlen)

                # model = Word2Vec.load('Datasets/emb.model')

                data_source_aux = []
                data_target_aux = []

                embedding_matrix = np.zeros((vocabulary_size, embedding_size))

                for word in tokenizer.word_index:
                    i = tokenizer.word_index[word]
                    if i >= vocabulary_size:
                        break

                    if '-' in word:
                        aux = word.split('-')
                        aux1 = aux[0]
                        aux2 = aux[1]

                        if aux1 not in model and aux2 not in model:
                            embedding_matrix[i] = np.zeros(embedding_size)
                        elif aux1 not in model:
                            embedding_matrix[i] = model[aux2]
                        else:
                            embedding_matrix[i] = model[aux1]
                    else:
                        if word in model:
                            embedding_matrix[i] = model[word]
                        else:
                            embedding_matrix[i] = np.zeros(embedding_size)

                '''
                source_emb = []
                for text in data_source_aux:
                    text_emb = []
                    for word in text:
                        if '_' in word:
                            aux = word.split('-')
            
                            aux2 = []
                            if aux[0] in model.wv:
                                aux2 = model.wv[aux[0]]
            
                            aux3 = []
                            if aux[1] in model.wv:
                                aux3 = model.wv[aux[1]]
            
                            if len(aux2) == 0:
                                if len(aux3) > 0:
                                    text_emb.append(aux3)
                                else:
                                    text_emb.append(model.wv['a'])
                            elif len(aux3) == 0:
                                if len(aux2) > 0:
                                    text_emb.append(aux2)
                                else:
                                    text_emb.append(model.wv['a'])
                            else:
                                text_emb.append(distance.euclidean(aux2, aux3))
                        else:
                            if word in model.wv:
                                text_emb.append(model.wv[word])
                            else:
                                text_emb.append(model.wv['a'])
                    source_emb.append(text_emb)
            
                target_emb = []
                for text in data_target_aux:
                    text_emb = []
                    for word in text:
                        if '_' in word:
                            aux = word.split('-')
            
                            aux2 = []
                            if aux[0] in model.wv:
                                aux2 = model.wv[aux[0]]
            
                            aux3 = []
                            if aux[1] in model.wv:
                                aux3 = model.wv[aux[1]]
            
                            if len(aux2) == 0:
                                if len(aux3) > 0:
                                    text_emb.append(aux3)
                                else:
                                    text_emb.append(model.wv['a'])
                            elif len(aux3) == 0:
                                if len(aux2) > 0:
                                    text_emb.append(aux2)
                                else:
                                    text_emb.append(model.wv['a'])
                            else:
                                text_emb.append(distance.euclidean(aux2, aux3))
                        else:
                            if word in model.wv:
                                text_emb.append(model.wv[word])
                            else:
                                text_emb.append(model.wv['a'])
                    target_emb.append(text_emb)
                '''
                convl = neural_networks.create_conv_model(vocabulary_size, embedding_size, embedding_matrix)
                # convl = neural_networks.convL(vocabulary_size, embedding_size, embedding_matrix)
                y_train = np_utils.to_categorical(label_source, 2)
                y_test = np_utils.to_categorical(label_target, 2)

                convl.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
                convl.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

                scores = convl.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
                print("%s: %.2f%%" % (convl.metrics_names[1], scores[1] * 100))

            print("src", src, "tgt", tgt, "num_layers", num_layers, 'pos', pos, 'text_representation', text_rep, "n clusters", n)
            print("\n-------------------------------------------------------------\n")