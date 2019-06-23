import pickle

from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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

# book = xlsxwriter.Workbook('Sheets/baseline1.xls')
# sheet = book.add_worksheet('1')

i = 1
for src in ['books', 'electronics', 'dvd', 'kitchen']:
    with open('Datasets/dataset_' + src, 'rb') as fp:
        dataset_source = pickle.load(fp)

    x_train, x_test, y_train, y_test = train_test_split(dataset_source.docs, dataset_source.labels, test_size=0.3,
                                                               random_state=42)

    # print(x_train)
    # print(y_train)

    cv = TfidfVectorizer(smooth_idf=True, norm='l1', max_features=5000)
    x_train = cv.fit_transform(x_train)
    x_test = cv.fit_transform(x_test)

    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)

    model = Sequential()
    model.add(Dense(1000, input_shape=(5000,)))
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

    model.fit(x_train, y_train, batch_size=batch_size, epochs=2, verbose=0)
    scores = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
    print(src)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # sheet.write(i, 0, src)
    # sheet.write(0, j, tgt)
    # sheet.write(i, j, scores[1] * 100)

i += 1
# book.close()
