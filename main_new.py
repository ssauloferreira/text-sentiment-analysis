import pickle
import spacy

from progress.bar import Bar

# ======================= PARAMETERS ==============================
pos = '1111'
minimum_tf = 5
# ========================= MODELS ================================
nlp = spacy.load("en_core_web_sm")
pos_mapping = {
    "ADJ": "a", "ADJ": "a", "ADJ": "a", "VERB": "v",
    "VERB": "v", "VERB": "v", "VERB": "v", "VERB": "v",
    "VERB": "v", "NOUN": "n", "NOUN": "n", "NOUN": "n",
    "NOUN": "n", "ADV": "r", "ADV": "r", "ADV": "r"
}
# =================================================================


def prepare_environ():
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('sentiwordnet')
    nltk.download('wordnet')


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

    with open('Datasets/dataset_electronics', 'rb') as fp:
        dataset = pickle.load(fp)
    datasets['electronics'] = dataset.docs
    labels['electronics'] = dataset.labels

    with open('Datasets/dataset_kitchen', 'rb') as fp:
        dataset = pickle.load(fp)
    datasets['kitchen'] = dataset.docs
    labels['kitchen'] = dataset.labels

    return datasets, labels


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
                     if term_frequency[key] > minimum_tf]

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
                continue

            token = token + "_" + pos
            new_text.append(token)

        new_dataset.append(new_text)
        bar.next()

    bar.finish()
    return new_dataset


if __name__ == "__main__":
    # prepare_environ()
    datasets, labels = load_amazon_data()

    for key in datasets.keys():
        datasets[key] = preprocess(dataset=datasets[key],
                                   pos_tags=pos,
                                   minimum_tf=5,
                                   label=key)
    
    print(datasets["books"])
