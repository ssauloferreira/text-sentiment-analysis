import pickle

import spacy
import xlsxwriter
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2

nlp = spacy.load('en_core_web_sm')
nfeature = 8000
n = 500
src = 'books'
tgt = 'electronics'
maxlen = 500
batch_size = 32
filters = 250
kernel_size = 5
hidden_dims = 250
epochs = 3
nb_epoch_t = 50

book = xlsxwriter.Workbook('Sheets/baseline2.xls')
sheet = book.add_worksheet('1')

i = 1


def get_vocab(dataset):
    vocab = {}
    for text in dataset:
        doc = nlp(text)

        for word in doc:
            if word.lemma_ != '-PRON-':
                vocab[word.lemma_] = 0

    return vocab.keys()



for src in ['books', 'electronics', 'dvd', 'kitchen']:
    j = 1
    for tgt in ['books', 'electronics', 'dvd', 'kitchen']:
        if src != tgt:
            with open('Datasets/dataset_' + src, 'rb') as fp:
                dataset_source = pickle.load(fp)

            with open('Datasets/dataset_' + tgt, 'rb') as fp:
                dataset_target = pickle.load(fp)

            vocabulary_src = get_vocab(dataset_source.docs)
            vocabulary_tgt = get_vocab(dataset_target.docs)

            vocabulary = []
            for word in vocabulary_src:
                if word in vocabulary_tgt:
                    vocabulary.append(word)

            cv_source = CountVectorizer(max_df=0.95, min_df=2, vocabulary=vocabulary)
            x_source = cv_source.fit_transform(dataset_source.docs)

            chi_stats, p_vals = chi2(x_source, dataset_source.labels)
            chi_res = sorted(list(zip(cv_source.get_feature_names(), chi_stats)),
                             key=lambda x: x[1], reverse=True)[0:5000]

            features = []
            for chi in chi_res:
                features.append(chi[0])

            cv = TfidfVectorizer(smooth_idf=True, norm='l1', vocabulary=features)
            x_train = cv.fit_transform(dataset_source.docs)
            x_test = cv.fit_transform(dataset_target.docs)

            y_train = np_utils.to_categorical(dataset_source.labels, 2)
            y_test = np_utils.to_categorical(dataset_target.labels, 2)

            model = Sequential()
            model.add(Dense(1000, input_shape=(len(features),)))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(500))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(50))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(2))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
            scores = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

            sheet.write(i, 0, src)
            sheet.write(0, j, tgt)
            sheet.write(i, j, scores[1] * 100)
        j += 1
    i += 1
book.close()
