import neural_networks
import numpy as np

matrix = np.zeros((8000, 300))
x_train = np.zeros((2000, 100))
y_train = np.zeros((2000, 2))

model = neural_networks.create_conv_model(8000, 300, embedding_matrix=matrix)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=0)
model.summary()