
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

import matplotlib.pyplot as plt

P = 550
P_t = 1000
Ni = 100
Nh = 50
No = 10

sigma_o = 2.5

nb_epoch = 120

X_train = np.random.normal(0,1.0,(P, Ni))
X_test = np.random.normal(0,1.0,(P_t, Ni))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

W0 = np.random.normal(0,1.0,(Ni,No))

Y_train = np.dot(X_train,W0) + np.random.normal(0,sigma_o,(P, No))
Y_test = np.dot(X_test,W0) + np.random.normal(0,sigma_o,(P_t, No))

model = Sequential()
model.add(Dense(Nh, input_shape=(Ni,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(Nh))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(No))
#model.add(Activation('softmax'))

model.summary()

sgd = SGD(lr=.001)
model.compile(loss='mse',
              optimizer=sgd)

history = model.fit(X_train, Y_train,
                    batch_size=1, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))


epoch = np.linspace(0, nb_epoch, nb_epoch)

fig, ax = plt.subplots(1)
line1, = ax.plot(epoch, history.history['loss'], '--', linewidth=2,
                 label='Train loss')
line2, = ax.plot(epoch, history.history['val_loss'], '--', linewidth=2,
                 label='Test loss')
ax.legend(loc='upper right')
plt.show()

