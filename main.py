import pickle
import copy
import gensim
import numpy as np
import logging

from gensim.models import Word2Vec
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.tree import tree
from flair.embeddings import FlairEmbeddings, WordEmbeddings
from flair.data import Sentence

import torch
import neural_networks
from pre_processing import to_process, get_vocabulary, get_senti_representation

logger = logging.getLogger()


def to_string(lists):
    new_docs = []

    for item in lists:
        text = ""
        for word in item:
            text = text + " " + word
        new_docs.append(text)

    return new_docs


embedding_models = {
                    # "bert": BertEmbeddings(),
                    # "elmo": ELMoEmbeddings(),
                    "flair": FlairEmbeddings('news-backward-fast'),
                    "default": WordEmbeddings('glove')
                }
classif = "mlp"
vocabulary_size = 10000
maxlen = 50
embedding_size = 1024
text_rep = 'flair'
embedding_model = 'flair'
pos = '1111'
num_layers = 200
nfeature = 8000
n = 200
maxlen = 50
batch_size = 64
filters = 250
kernel_size = 5
hidden_dims = 250
epochs = 5
nb_epoch_t = 50
_ = None


def get_bin(x, n): return format(x, 'b').zfill(n)


def generatePredictionData(dataset, max_length,
                           emb_size, embedding_model):
    x_batch = []
    for i, text in enumerate(dataset):
        print(f"{i}/{len(dataset)}")
        my_sent = text
        sentence = Sentence(my_sent)
        embedding_model.embed(sentence)

        x = []
        for token in sentence:
            x.append(token.embedding.cpu().detach().numpy())
            if len(x) == max_length:
                break

        while len(x) < max_length:
            x.append(np.zeros(emb_size))

        x_batch.append(x)
    return x_batch


# -------------------------- preprocessing ------------------------------------
#print("\npreprocessing=====================================\n")

datasets = {}
labels = {}

with open('Datasets/dataset_books', 'rb') as fp:
    dataset = pickle.load(fp)
data = to_process(dataset.docs[950:1050], pos, 3)
datasets['books'] = data
labels['books'] = dataset.labels[950:1050]

with open('Datasets/dataset_dvd', 'rb') as fp:
    dataset = pickle.load(fp)
data = to_process(dataset.docs[950:1050], pos, 3)
datasets['dvd'] = data
labels['dvd'] = dataset.labels[950:1050]
print(labels['dvd'])

# with open('Datasets/dataset_electronics', 'rb') as fp:
#     dataset = pickle.load(fp)
# data = to_process(dataset.docs, pos, 3)
# datasets['electronics'] = data
# labels['electronics'] = dataset.labels

# with open('Datasets/dataset_kitchen', 'rb') as fp:
#     dataset = pickle.load(fp)
# data = to_process(dataset.docs, pos, 3)
# datasets['kitchen'] = data
# labels['kitchen'] = dataset.labels

for src in ['books', 'dvd', 'electronics', 'kitchen']:
    for tgt in ['books', 'dvd', 'electronics', 'kitchen']:
        if src != tgt:
            print("[PROGRESS] |\tPreprocessing\t|\tClustering Features\t|\tConnecting Features\t|")

            data_source = datasets[src]
            label_source = labels[src]
            data_target = datasets[tgt]
            label_target = labels[tgt]

            # --------------------- clustering --------------------------------
            #print("[PROGRESS] Clustering features")

            vocabulary_source = get_vocabulary(data_source)
            #print('[PROGRESS] Vocabulary source:', len(vocabulary_source))
            vocab_source, scores_source, dicti_source = \
                get_senti_representation(vocabulary_source, True)

            vocabulary_target = get_vocabulary(data_target)
            #print('[PROGRESS] Vocabulary target:', len(vocabulary_target))
            vocab_target, scores_target, dicti_target = \
                get_senti_representation(vocabulary_target, True)
            print("           |======== OK ========|", end="")

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
            print("============== OK =============|", end="")

            # ----------------- feature selection -----------------------------
            common = []

            #print("[PROGRESS] Number of sentiment features source:",
                        # len(vocab_source))
            #print("[PROGRESS] Number of sentiment features target:",
                        # len(vocab_target))

            for feature in vocab_source:
                if feature in vocab_target:
                    common.append(feature)

            features_source = common
            #print("[PROGRESS] Number of common features: ", len(common))

            # ----------------- agrupando clusters ----------------------------
            #print("Linking the most similar clusters")
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

            # ------------------- calculating ALSENT --------------------------
            #print("[PROGRESS] Connecting features")

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
                except ZeroDivisionError:
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
                        except ZeroDivisionError:
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
                        aux = "".join([weighted_source[x][j][0][:-2],
                                       '-', weighted_target[y][j][0][:-2]])

                        grouped_features[weighted_source[x][j][0]] = aux
                        grouped_features[weighted_target[y][j][0]] = aux
            print("============== OK =============|\n")

            # --------------------- feature replacement -----------------------
            data_source_aux = copy.deepcopy(data_source)
            for i in range(len(data_source_aux)):
                for j in range(len(data_source_aux[i])):
                    if data_source_aux[i][j] in grouped_features:
                        data_source_aux[i][j] = \
                            grouped_features[data_source_aux[i][j]]

            data_target_aux = copy.deepcopy(data_target)
            for i in range(len(data_target_aux)):
                for j in range(len(data_target_aux[i])):
                    if data_target_aux[i][j] in grouped_features:
                        data_target_aux[i][j] = \
                            grouped_features[data_target_aux[i][j]]

            #print("[PROGRESS] Data has been already formatted.")
            if text_rep == 'tf-idf':
                # ------------------ feature selection ------------------------
                features_linked = list(dict.fromkeys(grouped_features.values()))
                features = features_linked + features_source

                cv_source = CountVectorizer(max_df=0.95, min_df=2,
                                            vocabulary=features)
                x_source = cv_source.fit_transform(to_string(data_source_aux))

                chi_stats, p_vals = chi2(x_source, label_source)
                chi_res = sorted(list(zip(cv_source.get_feature_names(),
                                          chi_stats)),
                                 key=lambda x: x[1], reverse=True)[0:1500]

                features = []
                for chi in chi_res:
                    features.append(chi[0])

                num_words = len(features)

                # -------------------- tf-idf ---------------------------------
                cv = TfidfVectorizer(smooth_idf=True,
                                     norm='l1',
                                     vocabulary=features)

                x_train = cv.fit_transform(to_string(data_source_aux))
                x_test = cv.transform(to_string(data_target_aux))

                #  ------------------ classifying  ----------------------------
                if classif == "logistic regression":
                    classifier = LogisticRegression()
                    classifier.fit(x_train, label_source)
                    predict = classifier.predict(x_test)

                    precision = f1_score(label_target, predict,
                                         average='binary')
                    #print('Precision:', precision)
                    accuracy = accuracy_score(label_target, predict)
                    #print('Accuracy: ', accuracy)
                    recall = recall_score(label_target, predict,
                                          average='binary')
                    #print('Recall: ', recall)
                    confMatrix = confusion_matrix(label_target, predict)
                    #print('Confusion matrix: \n', confMatrix)
                    #print('\n')
                    #print(src, tgt, ": ", accuracy*100)

                if classif == "deicion tree":
                    clf = tree.DecisionTreeClassifier()
                    clf.fit(x_train, label_source)
                    predict = clf.predict(x_test)

                    precision = f1_score(label_target, predict,
                                         average='binary')
                    #print('Precision:', precision)
                    accuracy = accuracy_score(label_target, predict)
                    #print('Accuracy: ', accuracy)
                    recall = recall_score(label_target, predict,
                                          average='binary')
                    #print('Recall: ', recall)
                    confMatrix = confusion_matrix(label_target, predict)
                    #print('Confusion matrix: \n', confMatrix)
                    #print('\n')
                    #print(src, tgt, ": ", accuracy*100)

                if classif == "mlp":
                    y_train = np_utils.to_categorical(label_target, 2)
                    y_test = np_utils.to_categorical(label_source, 2)

                    model = neural_networks.mlp(input_shape=(num_words,),
                                                num_layers=num_layers)
                    # model.summary()
                    model.compile(loss='categorical_crossentropy',
                                  optimizer='adam', metrics=["accuracy"])

                    model.fit(x_train,
                              y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=1)
                    scores = model.evaluate(x_test,
                                            y_test,
                                            verbose=0,
                                            batch_size=batch_size)
                    print("src", src, "tgt", tgt, "num_layers",
                                num_layers, 'pos', pos)

                    print(src, tgt, "%s: %.2f%%" % (model.metrics_names[1],
                                                    scores[1] * 100))

            elif text_rep == 'embeddings':
                print("[CLASSIFICATION] |\tProcessing embeddings\t|\tGen. embedding matrix\t|\tFitting neural network\t|")
                embedding = embedding_models.get(embedding_model,
                                                 embedding_models["default"])
                model = {}

                for i, text in enumerate(data_source_aux):
                    for j, word in enumerate(text):
                        if '_' in word:
                            text[j] = word[:-2]

                for i, text in enumerate(data_target_aux):
                    for j, word in enumerate(text):
                        if '_' in word:
                            text[j] = word[:-2]

                data_source_str = to_string(data_source_aux)
                data_target_str = to_string(data_target_aux)

                #print("Generating embedding matrix")
                # for text_source, text_target in zip(data_source_str, 
                #                                     data_target_str):
                #     sentence = Sentence(text_source)
                #     print(sentence)
                #     embedding.embed(sentence)
                #     for token in sentence:
                #         word = str(token).split(" ")[-1]
                #         model[word] = token.embedding.tolist()

                #     sentence = Sentence(text_target)
                #     embedding.embed(sentence)
                #     for token in sentence:
                #         word = str(token).split(" ")[-1]
                #         model[word] = token.embedding.tolist()
                print("                 |============= OK =============|", end="")

                #print("Tokenizing it all")
                tokenizer = Tokenizer(num_words=vocabulary_size)
                tokenizer.fit_on_texts(data_source_aux)

                sequences_src = tokenizer.texts_to_sequences(data_source_aux)
                sequences_tgt = tokenizer.texts_to_sequences(data_target_aux)

                x_train = pad_sequences(sequences_src,
                                        padding='post',
                                        maxlen=maxlen)
                x_test = pad_sequences(sequences_tgt,
                                       padding='post',
                                       maxlen=maxlen)

                #print("Transform into embedding vectors")
                embedding_matrix = np.zeros((vocabulary_size, embedding_size))

                for word in tokenizer.word_index:
                    i = tokenizer.word_index[word]
                    if i >= vocabulary_size:
                        break

                    sentence = Sentence(word.split("-")[0])
                    embedding.embed(sentence)
                    for token in sentence:
                        word = str(token).split(" ")[-1]
                        emb = token.embedding
                        embedding_matrix[i] = emb
                        print(f"{word}: {i}")
                print(embedding_matrix)

                print("============= OK ==============|", end="")

                convl = neural_networks.create_conv_model(vocabulary_size,
                                                          embedding_size,
                                                          embedding_matrix,
                                                          maxlen)
                y_train = np_utils.to_categorical(label_source, 2)
                y_test = np_utils.to_categorical(label_target, 2)

                convl.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=["accuracy"])
                convl.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs, verbose=0)
                print(convl.summary())
                print("============= OK ==============|\n")

                scores = convl.evaluate(x_test, y_test,
                                        verbose=0,
                                        batch_size=batch_size)
                print("[RESULT]")
                print("\t", src, tgt, "%s: %.2f%%" % (convl.metrics_names[1],
                                                      scores[1] * 100))

            elif text_rep == "flair":
                model = embedding_models.get(embedding_model,
                                             embedding_models["default"])

                for i, text in enumerate(data_source_aux):
                    for j, word in enumerate(text):
                        if '_' in word:
                            text[j] = word[:-2]
                        if '-' in word:
                            text[j] = word.split("-")[0]

                for i, text in enumerate(data_target_aux):
                    for j, word in enumerate(text):
                        if '_' in word:
                            text[j] = word[:-2]
                        if '-' in word:
                            text[j] = word.split("-")[0]

                data_source_str = to_string(data_source_aux)
                print(data_source_str)
                data_target_str = to_string(data_target_aux)

                x_train = generatePredictionData(dataset=data_source_str,
                                                 max_length=maxlen,
                                                 emb_size=embedding_size,
                                                 embedding_model=model)

                x_test = generatePredictionData(dataset=data_target_str,
                                                max_length=maxlen,
                                                emb_size=embedding_size,
                                                embedding_model=model)
                print(x_train)

                x_train = np.array(x_train)
                x_test = np.array(x_test)

                y_train = np_utils.to_categorical(label_source, 2)
                y_test = np_utils.to_categorical(label_target, 2)

                convl = neural_networks.convL(input_shape=(maxlen,
                                                           embedding_size))
                convl.compile(loss='categorical_crossentropy',
                              optimizer='adam', metrics=["accuracy"])
                print(convl.summary())

                convl.fit(x_train,
                          y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1)

                scores = convl.evaluate(x_test,
                                        y_test,
                                        verbose=1,
                                        batch_size=batch_size)
                print(scores)
                print(y_test)

                print(src, tgt, "%s: %.2f%%" % (convl.metrics_names[1],
                                                scores[1] * 100))
