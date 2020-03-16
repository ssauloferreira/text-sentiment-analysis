from keras import Sequential
from keras.layers import Conv1D, LSTM, Dropout, Dense,\
                        Activation, Embedding, MaxPooling1D, \
                        SpatialDropout1D
from keras.regularizers import l2


def create_conv_model(vocabulary_size, embedding_size,
                      embedding_matrix, maxlen):
    model_conv = Sequential()
    model_conv.add(Embedding(vocabulary_size, embedding_size,
                             weights=[embedding_matrix], input_length=maxlen,
                             trainable=False))
    model_conv.add(Conv1D(250, 5, activation='relu',
                          padding='valid', strides=1))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(LSTM(100))
    model_conv.add(Dropout(0.5))
    model_conv.add(Dense(2, activation='sigmoid', kernel_regularizer=l2(3)))
    return model_conv


def convL(input_shape):
    model_conv = Sequential()
    model_conv.add(Conv1D(250, 5, activation='relu',
                   padding='valid', strides=1, input_shape=input_shape))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(LSTM(100))
    model_conv.add(Dropout(0.5))
    model_conv.add(Dense(2, activation='sigmoid', kernel_regularizer=l2(3)))
    return model_conv

    model = Sequential()

    for l in classification_layers:
        model.add(l)

    return model


def mlp(input_shape, num_layers=1000):
    a = num_layers
    b = int(num_layers/2)
    c = int(num_layers/10)

    model = Sequential()
    model.add(Dense(a, input_shape=input_shape))
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


def lstm(vocabulary_size, embedding_size, embedding_matrix, maxlen):
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_size,
                        weights=[embedding_matrix], input_length=maxlen,
                        trainable=False))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    return model
