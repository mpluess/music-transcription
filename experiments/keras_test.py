from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

data_dim = 20
nb_classes = 4

model = Sequential()

# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, input_dim=data_dim, kernel_initializer='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_initializer='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, kernel_initializer='uniform'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',  
              metrics=["accuracy"])

# generate dummy training data
x_train = np.random.random((1000, data_dim))
y_train = np.random.random((1000, nb_classes))

# generate dummy test data
x_test = np.random.random((100, data_dim))
y_test = np.random.random((100, nb_classes))

model.fit(x_train, y_train,
          epochs=20,
          batch_size=16)

score = model.evaluate(x_test, y_test, batch_size=16)
