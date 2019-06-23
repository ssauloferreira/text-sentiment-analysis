from keras import Sequential
from keras.layers import Conv1D, LSTM, Dropout, Dense, Activation, Embedding
from keras.regularizers import l2


def convL(input_shape, embedding_size, input_len):
    filters = 250
    kernel_size = 5

    classification_layers = [
        Embedding(input_dim=input_shape, output_dim=embedding_size, input_length=input_len),
        Conv1D(input_shape=(input_len, embedding_size), filters=filters, kernel_size=kernel_size, padding='valid',
               activation='relu',
               strides=1),
        LSTM(100),
        Dropout(0.25),
        Dense(2, kernel_regularizer=l2(3)),
        Activation('sigmoid')

    ]

    model = Sequential()

    for l in classification_layers:
        model.add(l)

    return model


def mlp(input_shape, num_layers=1000):
    a = num_layers
    b = int(num_layers/2)
    c = int(num_layers/10)

    model = Sequential()
    model.add(Dense(a, input_shape=(input_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(b))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(c))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model
