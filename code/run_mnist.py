
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

import matplotlib.pyplot as plt

P = 100;

batch_size = P

nb_classes = 10
nb_epoch = 1000

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255



print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train[1:P, :]
Y_train = Y_train[1:P, :]

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(10))
#model.add(Activation('softmax'))

model.summary()

model.compile(loss='mse',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))


epoch = np.linspace(0, nb_epoch, nb_epoch)

fig, ax = plt.subplots(1)
line1, = ax.plot(epoch, history.history['loss'], '--', linewidth=2,
                 label='Train loss')
line2, = ax.plot(epoch, history.history['val_loss'], '--', linewidth=2,
                 label='Test loss')
ax.legend(loc='upper right')
plt.show()

