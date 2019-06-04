import pickle

from keras import Sequential
from keras.layers import Conv1D, LSTM, Dropout, Dense, Activation, Flatten
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.utils import np_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from pre_processing import to_process


def to_string(lists):
    new_docs = []

    for item in lists:
        text = ""
        for word in item:
            text = text + " " + word
        new_docs.append(text)

    return new_docs


with open('Datasets/dataset_books', 'rb') as fp:
    dataset_source = pickle.load(fp)

data_source, data_target, label_source, label_target = train_test_split(dataset_source.docs, dataset_source.labels,
                                                                        test_size=0.3, random_state=42)
data_source = to_process(data_source, '6', 0)
data_target = to_process(data_target, '6', 0)

print(len(data_source))
print(len(data_target))

cv = TfidfVectorizer(smooth_idf=True, norm='l1', max_features=3000)
x_train = cv.fit_transform(to_string(data_source))
x_test = cv.fit_transform(to_string(data_target))

label_target = np_utils.to_categorical(label_target, 2)
label_source = np_utils.to_categorical(label_source, 2)

maxlen = 500
batch_size = 32
filters = 250
kernel_size = 5
hidden_dims = 250
epochs = 3
nb_epoch_t = 50
_ = None

print(x_train.shape)
x_train = x_train.reshape((1400, 1, 3000))
print(x_train.shape)

'''
classification_layers = [
    Conv1D(input_shape=(3000, 1), filters=filters, kernel_size=kernel_size, padding='valid', activation='relu',
           strides=1),
    LSTM(100),
    Dropout(0.25),
    Dense(1, kernel_regularizer=l2(3)),
    Activation('sigmoid')
]

print('Build model...')
model = Sequential()

for l in classification_layers:
    model.add(l)

rmsprop = RMSprop(lr=0.0005, decay=1e-6, rho=0.9)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
model.fit(x_train, label_source, batch_size=batch_size,
          epochs=nb_epoch_t, validation_split=0.3)
score = model.evaluate(x_test, label_target, batch_size=batch_size, verbose=1)

print('new dataset Test score:', score[0])
print('new dataset Test accuracy:', score[1])
'''

# Deep Layer Model building in Keras

# del model
model = Sequential()
model.add(Dense(1000, input_shape=(3000,)))
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

print(model.summary())

model.fit(x_train, label_source, batch_size=batch_size, epochs=epochs,verbose=1)

