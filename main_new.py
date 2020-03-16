import pickle
import spacy
import itertools
import numpy as np
import nltk
import math
import neural_networks

from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial import distance
from sklearn.cluster import KMeans
from nltk.corpus import sentiwordnet as swn
from progress.bar import Bar
from flair.embeddings import FlairEmbeddings, WordEmbeddings, BertEmbeddings, ELMoEmbeddings
from flair.data import Sentence

# ======================= PARAMETERS ==============================
pos = '1101'
minimum_tf = 5
n_cluster = 50
limit = 200
classifier = "lstm"
batch_size = 64
epochs = 10
num_layers = 200
max_words = 8000
embedding_model = "default"
max_length = 50
# ========================= MODELS ================================
nlp = spacy.load("en_core_web_sm")
pos_mapping = {
    "ADJ": "a", "VERB": "v", "NOUN": "n", "ADV": "r", "PROPN": "n"
}
negative_words = ['not', 'no', 'nothing', 'never']
negative_pos = ["v", "a"]
embedding_models = {
                    "bert": BertEmbeddings(),
                    # "elmo": ELMoEmbeddings(),
                    "flair": FlairEmbeddings('news-backward-fast'),
                    "default": WordEmbeddings('glove')
                }
# =================================================================


def prepare_environ():
    try:
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('sentiwordnet')
        nltk.download('wordnet')
    except Exception:
        print("[INFO] It wasn't necessary to download NLTK features.")


def load_amazon_data(limit=200):
    datasets = {}
    labels = {}

    with open('Datasets/dataset_books', 'rb') as fp:
        dataset = pickle.load(fp)
    datasets['books'] = dataset.docs[:limit]
    labels['books'] = dataset.labels[:limit]

    with open('Datasets/dataset_dvd', 'rb') as fp:
        dataset = pickle.load(fp)
    datasets['dvd'] = dataset.docs[:limit]
    labels['dvd'] = dataset.labels[:limit]

    # with open('Datasets/dataset_electronics', 'rb') as fp:
    #     dataset = pickle.load(fp)
    # datasets['electronics'] = dataset.docs[:limit]
    # labels['electronics'] = dataset.labels[:limit]

    # with open('Datasets/dataset_kitchen', 'rb') as fp:
    #     dataset = pickle.load(fp)
    # datasets['kitchen'] = dataset.docs[:limit]
    # labels['kitchen'] = dataset.labels[:limit]

    return datasets, labels


def to_string(lists):
    new_docs = []

    for item in lists:
        text = ""
        for word in item:
            text = text + " " + word
        new_docs.append(text)

    return new_docs


# input: list of text (unprocessed) or list of lists of strings (processed)
# ouput: dictionary | token and its frequency
def word_frequency(dataset):
    term_frequency = {}
    for text in dataset:
        if type(text) is str: 
            for word in nlp(text):
                token = word.lemma_.lower()
                count = term_frequency.get(token, 0)
                term_frequency[token] = count + 1
        else:
            for word in text:
                count = term_frequency.get(word, 0)
                term_frequency[word] = count + 1

    return term_frequency


# input: list of strings | unprocessed data
# output: list of lists of strings | preprocessed data
def preprocess(dataset, pos_tags, minimum_tf, label):
    bar = Bar(label, max=len(dataset))
    new_dataset = []
    pos_filter = []

    # Listing rare features
    term_frequency = word_frequency(dataset)
    rare_features = [key for key in term_frequency.keys()
                     if term_frequency[key] < minimum_tf]

    for i in range(4):
        if pos_tags[i] == '1':
            if i == 0:
                pos_filter.append('a')
            elif i == 1:
                pos_filter.append('v')
            elif i == 2:
                pos_filter.append('n')
            else:
                pos_filter.append('r')

    for text in dataset:
        new_text = []
        doc = nlp(text)
        negative = False

        for word in doc:
            pos = pos_mapping.get(word.pos_, False)
            token = word.lemma_.lower()

            if not pos or word.is_punct or word.is_space \
                    or token in rare_features or pos not in pos_filter:
                continue

            if token in negative_words:
                negative = True

            if negative and pos in negative_pos:
                token = token + ".not"
                negative = False

            token = token + "_" + pos
            new_text.append(token)

        new_dataset.append(new_text)
        bar.next()

    bar.finish()
    return new_dataset


# input: list of lists of strings | processed data
# ouput: list of strings | vocabulary
def get_vocabulary(dataset):
    vocab = {}

    for text in dataset:
        for word in text:
            vocab[word] = 0

    return vocab.keys()


# input: list of strings | vocabulary
# ouput: dictionary | word as keys, scores as items
def words_to_swn(vocabulary, pos_form=True, minimum_score=0.0):
    scores = {}

    for item in vocabulary:
        if item in scores:
            continue

        pos_word = item.split("_")
        if len(pos_word) == 3:
            continue

        word, pos = pos_word
        syns = list(swn.senti_synsets(word))

        neg_word = word.split(".")
        negative = False
        if len(neg_word) == 2:
            negative = True
            word = neg_word[0]

        if not syns:
            continue

        pos_score = []
        neg_score = []
        obj_score = []
        for syn in syns:
            if pos in syn.synset.name():
                pos_score.append(syn.pos_score())
                neg_score.append(syn.neg_score())
                obj_score.append(syn.obj_score())

        if pos_score:
            if negative:
                score = [round(np.mean(neg_score), 3),
                         round(np.mean(pos_score), 3),
                         round(np.mean(obj_score), 3)]
            else:
                score = [round(np.mean(pos_score), 3),
                         round(np.mean(neg_score), 3),
                         round(np.mean(obj_score), 3)]

            if score[0] > minimum_score or score[1] > minimum_score:
                word = word + ".not" if negative else word
                word = word + "_" + pos if pos_form else word
                scores[word] = score

    return scores


# input: dictionary | words and its scores
# output: two lists | clusters of words
def cluster_words(vocabulary_scores):
    words, scores = zip(*vocabulary_scores.items())
    clustering = KMeans(n_clusters=n_cluster, random_state=49)
    clustering.fit(scores)

    word_clusters = [[] for i in range(n_cluster)]
    score_clusters = [[] for i in range(n_cluster)]

    for i in range(len(words)):
        aux = clustering.labels_[i]
        word_clusters[aux].append(words[i])
        score_clusters[aux].append(scores[i])

    return word_clusters, score_clusters


def compute_tfidf(docs, vocabulary):
    def computeTF(word_dict, bag_of_words):
        tf_dict = {}
        bag_of_words_count = len(bag_of_words)
        for word, count in word_dict.items():
            tf_dict[word] = count / float(bag_of_words_count)
        return tf_dict

    def computeIDF(documents, vocabulary):
        N = len(documents)

        idf_dict = {}
        for word in vocabulary:
            df = 0
            for document in documents:
                if word in document:
                    df += 1
            idf_dict[word] = df

        for word, val in idf_dict.items():
            idf_dict[word] = math.log(N / float(val))
        return idf_dict

    term_frequency = word_frequency(docs)
    tf = computeTF(term_frequency, vocabulary)
    idf = computeIDF(docs, vocabulary)

    tfidf_dict = {}
    for word, tf in tf.items():
        tfidf_dict[word] = tf * idf[word]

    return tfidf_dict


# input: list of lists of strings, dictionary
# ouput: list of strings
def replace_in_dataset(dataset, grouped_features, inverse=True):
    linked = []
    for i, text in enumerate(dataset):
        for j, word in enumerate(text):
            if word in grouped_features:
                if inverse:
                    aux = "".join([grouped_features[word],
                                   "-", word])
                    dataset[i][j] = aux
                    linked.append(aux)
                else:
                    aux = "".join([word, "-",
                                   grouped_features[word]])
                    dataset[i][j] = aux
                    linked.append(aux)
    return linked


def feature_selection(features, data, labels):
    cv_source = CountVectorizer(max_df=0.95, min_df=2,
                                vocabulary=features)
    x_source = cv_source.fit_transform(to_string(data))

    chi_stats, p_vals = chi2(x_source, labels)
    chi_res = sorted(list(zip(cv_source.get_feature_names(),
                              chi_stats)),
                     key=lambda x: x[1], reverse=True)[0:max_words]

    features = []
    for chi in chi_res:
        features.append(chi[0])

    return features


def tfidf_it(data_src, data_tgt, features):
    cv = TfidfVectorizer(smooth_idf=True,
                         norm='l1',
                         vocabulary=features)

    x_train = cv.fit_transform(to_string(data_src))
    x_test = cv.transform(to_string(data_tgt))

    return x_train, x_test


def fix_word_format(data):
    for i, text in enumerate(data):
        for j, word in enumerate(text):
            if word == "-pron-_v":
                data[i][j] = ""
                continue
            aux = ""
            if "-" in word:
                word1, word2 = word.split("-")
                word1 = word1.split("_")[0]
                word2 = word2.split("_")[0]
                if "." in word1:
                    word1 = word1.split(".")[0] + "_not"
                if "." in word2:
                    word2 = word2.split(".")[0] + "_not"
                aux = word1 + "_" + word2
                data[i][j] = aux
            else:
                aux = word.split("_")[0]
                if "." in aux:
                    aux = aux.split(".")[0] + "_not"
                data[i][j] = aux


if __name__ == "__main__":
    prepare_environ()
    limit = input("Number of tests: ")
    if limit == "":
        limit = 2000
    datasets, labels = load_amazon_data(limit=int(limit))

    print(f"[PROCESS] Preprocessing data. {len(datasets)} datasets found.")
    for key in datasets.keys():
        datasets[key] = preprocess(dataset=datasets[key],
                                   pos_tags=pos,
                                   minimum_tf=5,
                                   label=key)

    for source, target in itertools.combinations(datasets.keys(), 2):
        data_source = datasets[source]
        labels_source = labels[source]
        data_target = datasets[target]
        labels_target = labels[target]

        vocabulary_source = get_vocabulary(data_source)
        vocabulary_target = get_vocabulary(data_target)

        tfidf_score_src = compute_tfidf(docs=data_source,
                                        vocabulary=vocabulary_source)
        tfidf_score_tgt = compute_tfidf(docs=data_target,
                                        vocabulary=vocabulary_target)

        print("[PROCESS] Getting SWN scores from words.")
        vocabscore_src = words_to_swn(vocabulary_source)
        vocabscore_tgt = words_to_swn(vocabulary_target)

        vocab_score = {}
        vocab_score.update(vocabscore_src)
        vocab_score.update(vocabscore_tgt)

        print("[PROCESS] Clustering features by its scores.")
        wordcluster_src, scorecluster_src = cluster_words(vocabscore_src)
        wordcluster_tgt, scorecluster_tgt = cluster_words(vocabscore_tgt)

        common_features = set(vocabscore_src.keys()) \
            & set(vocabscore_tgt.keys())

        print("[PROCESS] Connecting clusters by common features.")
        grouped_clusters = {}
        for (i, cluster_a), (j, cluster_b) in itertools\
                .product(enumerate(wordcluster_src),
                         enumerate(wordcluster_tgt)):
            similariy = len(set(cluster_a) & set(cluster_b))
            current = grouped_clusters.get(i, (0, -1))

            grouped_clusters[i] = (similariy, j) \
                if similariy > current[0] else current

        print("[PROCESS] Connecting features")
        grouped_clusters = [(key, value[1]) for key, value
                            in grouped_clusters.items()]
        grouped_features = {}

        for i, j in grouped_clusters:
            if j == -1:
                continue
            for l, word_a in enumerate(wordcluster_src[i]):
                if word_a in common_features:
                    continue
                dist = math.inf
                index = -1

                for k, word_b in enumerate(wordcluster_tgt[j]):
                    if word_b in common_features and word_b not in grouped_features:
                        continue
                    try:
                        temp_dist = distance.euclidean(tfidf_score_src[word_a],
                                                       tfidf_score_tgt[word_b])
                        if temp_dist < dist:
                            dist = temp_dist
                            index = k
                    except Exception as ex:
                        print(ex)
                        pass

                if dist != math.inf:
                    grouped_features[word_a] = wordcluster_tgt[j][index]
                    grouped_features[wordcluster_tgt[j][index]] = word_a

        linked_features = replace_in_dataset(data_source,
                                             grouped_features,
                                             False)
        linked_features += replace_in_dataset(data_target,
                                              grouped_features,
                                              True)
        linked_features = list(dict.fromkeys(linked_features))

        print("[PROCRESS] Classifying...")
        if classifier == "mlp":
            features_selected = feature_selection(linked_features +
                                                  list(common_features),
                                                  data_source,
                                                  labels_source)

            x_train, x_test = tfidf_it(data_source, data_target,
                                       features_selected)
            y_train = np_utils.to_categorical(labels_source, 2)
            y_test = np_utils.to_categorical(labels_target, 2)

            model = neural_networks.mlp(input_shape=(max_words,),
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

            print(source, target, "%s: %.2f%%" % (model.metrics_names[1],
                                                  scores[1] * 100))
        else:
            embedding = embedding_models.get(embedding_model,
                                             embedding_models["default"])
            model = {}
            fix_word_format(data_source)
            fix_word_format(data_target)

            data_source_str = to_string(data_source)
            data_target_str = to_string(data_target)
            # print(data_source_str)

            # print(data_source_str)

            tokenizer = Tokenizer(num_words=max_words)
            tokenizer.fit_on_texts(data_source_str)

            sequences_src = tokenizer.texts_to_sequences(data_source_str)
            sequences_tgt = tokenizer.texts_to_sequences(data_target_str)

            x_train = pad_sequences(sequences_src,
                                    padding='post',
                                    maxlen=max_length)
            x_test = pad_sequences(sequences_tgt,
                                   padding='post',
                                   maxlen=max_length)

            aux = Sentence("default")
            embedding.embed(aux)
            for token in aux:
                embedding_size = len(token.embedding.tolist())

            print("[PROCESS] Transform into embedding vectors")
            embedding_matrix = np.zeros((max_words, embedding_size))

            for word in tokenizer.word_index:
                i = tokenizer.word_index[word]
                if i >= max_words:
                    break

                sentence = Sentence(word)
                embedding.embed(sentence)
                for token in sentence:
                    word = str(token).split(" ")[-1]
                    emb = token.embedding
                    embedding_matrix[i] = emb
            if classifier == "convl":
                network = neural_networks.create_conv_model(max_words,
                                                            embedding_size,
                                                            embedding_matrix,
                                                            max_length)
            elif classifier == "lstm":
                network = neural_networks.lstm(max_words,
                                               embedding_size,
                                               embedding_matrix,
                                               max_length)
            y_train = np_utils.to_categorical(labels_source, 2)
            y_test = np_utils.to_categorical(labels_target, 2)

            network.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=["accuracy"])
            network.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs, verbose=1)
            # print(network.summary())

            scores = network.evaluate(x_test, y_test,
                                      verbose=1,
                                      batch_size=batch_size)
            print("[RESULT]")
            print("\t", source, target, "%s: %.2f%%" % (network.metrics_names[1],
                                                        scores[1] * 100))
