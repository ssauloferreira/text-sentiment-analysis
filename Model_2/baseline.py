import pickle

import xlsxwriter
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, recall_score, accuracy_score

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

for src in ['books', 'dvd', 'electronics', 'kitchen']:
    for tgt in ['books', 'dvd', 'electronics', 'kitchen']:
        if src != tgt:
            with open('Datasets/dataset_' + src, 'rb') as fp:
                dataset_source = pickle.load(fp)

            with open('Datasets/dataset_' + tgt, 'rb') as fp:
                dataset_target = pickle.load(fp)

            cv = TfidfVectorizer(smooth_idf=True, norm='l1', max_features=2000)
            x_train = cv.fit_transform(dataset_source.docs)
            x_test = cv.transform(dataset_target.docs)

            y_train = np_utils.to_categorical(dataset_source.labels, 2)
            y_test = np_utils.to_categorical(dataset_target.labels, 2)

            model = Sequential()
            model.add(Dense(1000, input_shape=(2000,)))
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

            print(src, tgt)
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
            scores = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))