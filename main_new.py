import pickle
import spacy
import itertools
import numpy as np
import nltk
import time

from scipy.spatial import distance
from sklearn.cluster import KMeans
from nltk.corpus import sentiwordnet as swn
from progress.bar import Bar

# ======================= PARAMETERS ==============================
pos = '1111'
minimum_tf = 5
n_cluster = 100
limit = 200
# ========================= MODELS ================================
nlp = spacy.load("en_core_web_sm")
pos_mapping = {
    "ADJ": "a", "VERB": "v", "NOUN": "n", "ADV": "r", "PROPN": "n"
}
negative_words = ['not', 'no', 'nothing', 'never']
negative_pos = ["v", "a"]
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


# input: list of strings | unprocessed data
# output: list of lists of strings | preprocessed data
def preprocess(dataset, pos_tags, minimum_tf, label):
    bar = Bar(label, max=len(dataset))
    new_dataset = []
    pos_filter = []

    # Listing rare features
    term_frequency = {}
    for text in dataset:
        for word in nlp(text):
            token = word.lemma_.lower()
            count = term_frequency.get(token, 0)
            term_frequency[token] = count + 1

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


if __name__ == "__main__":
    prepare_environ()
    datasets, labels = load_amazon_data(limit=1000)

    print(f"[PROCESS] Preprocessing data. {len(datasets)} datasets found.")
    for key in datasets.keys():
        datasets[key] = preprocess(dataset=datasets[key],
                                   pos_tags=pos,
                                   minimum_tf=5,
                                   label=key)

    for source, target in itertools.combinations(datasets.keys(), 2):
        data_source = datasets[source]
        label_source = labels[source]
        data_target = datasets[target]
        labels_target = labels[target]

        vocabulary_source = get_vocabulary(data_source)
        vocabulary_target = get_vocabulary(data_target)

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

        for group in grouped_clusters:
            print(group, grouped_clusters[group])

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
                similariy = 0
                index = -1

                for k, word_b in enumerate(wordcluster_tgt[j]):
                    if word_b in common_features:
                        continue
                    try:
                        sim = 1/distance.euclidean(scorecluster_src[i][l],
                                                   scorecluster_tgt[j][k])
                    except Exception:
                        pass
                    if sim > similariy:
                        similariy = sim
                        index = k

                grouped_features[word_a] = wordcluster_tgt[j][index]

        print(grouped_features)
