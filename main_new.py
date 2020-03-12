import pickle
import spacy
import itertools
import numpy as np
import nltk

from nltk.corpus import sentiwordnet as swn
from progress.bar import Bar

# ======================= PARAMETERS ==============================
pos = '1111'
minimum_tf = 5
# ========================= MODELS ================================
nlp = spacy.load("en_core_web_sm")
pos_mapping = {
    "ADJ": "a", "VERB": "v", "NOUN": "n", "ADV": "r", "PROPN": "n"
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


def load_amazon_data():
    datasets = {}
    labels = {}

    with open('Datasets/dataset_books', 'rb') as fp:
        dataset = pickle.load(fp)
    datasets['books'] = dataset.docs
    labels['books'] = dataset.labels

    with open('Datasets/dataset_dvd', 'rb') as fp:
        dataset = pickle.load(fp)
    datasets['dvd'] = dataset.docs
    labels['dvd'] = dataset.labels

    # with open('Datasets/dataset_electronics', 'rb') as fp:
    #     dataset = pickle.load(fp)
    # datasets['electronics'] = dataset.docs
    # labels['electronics'] = dataset.labels

    # with open('Datasets/dataset_kitchen', 'rb') as fp:
    #     dataset = pickle.load(fp)
    # datasets['kitchen'] = dataset.docs
    # labels['kitchen'] = dataset.labels

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

        for word in doc:
            pos = pos_mapping.get(word.pos_, False)
            token = word.lemma_.lower()

            if not pos or word.is_punct or word.is_space \
                    or token in rare_features or pos not in pos_filter:
                # print(f"{token} | pos: {pos} | punct: {word.is_punct} | space: {word.is_space} | rare: {term_frequency.get(token, False)} {token in rare_features}")
                continue

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

        word, pos = item.split("_")
        syns = list(swn.senti_synsets(word))

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
            score = [round(np.mean(pos_score), 3),
                     round(np.mean(neg_score), 3),
                     round(np.mean(obj_score), 3)]

            if score[0] > minimum_score or score[1] > minimum_score:
                word = word + "_" + pos if pos_form else word
                scores[word] = score

    return scores


if __name__ == "__main__":
    prepare_environ()
    datasets, labels = load_amazon_data()

    print(f"[PROCESS] Preprocessing data. {len(datasets)} datasets have found.")
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

        print(words_to_swn(vocabulary_source))
