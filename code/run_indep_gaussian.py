
from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import Callback
from keras import backend as K
from keras import initializations

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-rseed', type=int, default=0)

parser.add_argument('-numinputs', type=int, default=100)                                                                                             
parser.add_argument('-numoutputs', type=int, default=10)
parser.add_argument('-numhid', type=int, default=50)
parser.add_argument('-depth', type=int, default=1)

parser.add_argument('-snr', type=float, default=2.5)
parser.add_argument('-numsamples', type=int, default=100)

parser.add_argument('-lr', type=float, default=.001)
parser.add_argument('-epochs',     type=int, default=100)
parser.add_argument('-batchsize', type=int, default=-1)
parser.add_argument('-weightscale', type=float, default=1.)


parser.add_argument('-savefile', type=argparse.FileType('w'))

parser.add_argument('-showplot', action='store_true')
parser.add_argument('-saveplot', action='store_true')
parser.add_argument('-verbose', action='store_true')


settings = parser.parse_args(); 

import matplotlib

if not settings.showplot:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(settings.rseed)


class GaussianGeneralizationHistory(Callback):
    def __init__(self, W0, No, Ni, P, sigma_o):
        self.W0 = W0
        self.No = No
        self.Ni = Ni
        self.P  = P
        self.sigma_o = sigma_o

    def on_train_begin(self, logs={}):
        self.genhist = []

    def on_batch_end(self, batch, logs={}):
        # Forward prop to find current total input-output map
        W = self.model.predict(np.identity(self.Ni), self.Ni)
        
        # Compute loss
        dW = self.W0 - W
        gen_err = (1./self.No)*np.linalg.norm(dW)**2 + self.sigma_o**2
        
        self.genhist.append(gen_err)
        

scaled_normal_init = lambda shape, name=None: initializations.normal(shape, scale=settings.weightscale/np.sqrt(np.mean(shape)), name=name)


P = settings.numsamples
P_t = 10000
Ni = settings.numinputs
Nh = settings.numhid
No = settings.numoutputs

sigma_o = 1/settings.snr
nb_epoch = settings.epochs
if settings.batchsize > 0:
    batch_sz = settings.batchsize
else:
    batch_sz = P


X_train = np.random.normal(0,1.0,(P, Ni))
X_test = np.random.normal(0,1.0,(P_t, Ni))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

W0 = np.random.normal(0,1.0,(Ni,No))

Y_train = np.dot(X_train,W0) + np.random.normal(0,sigma_o,(P, No))
Y_test = np.dot(X_test,W0) + np.random.normal(0,sigma_o,(P_t, No))

model = Sequential()
if settings.depth > 0: # Deep model
    model.add(Dense(Nh, input_shape=(Ni,), bias=False, init=scaled_normal_init))
    
    for d in range(settings.depth-1):
        model.add(Dense(Nh, bias=False, init=scaled_normal_init))
        #model.add(Activation('relu'))
        #model.add(Dropout(0.2))
        
    model.add(Dense(No, bias=False, init=scaled_normal_init))
else: # Shallow model
    model.add(Dense(No, input_shape=(Ni,), bias=False, init=scaled_normal_init))

if settings.verbose:
    model.summary()
                       
genhist = GaussianGeneralizationHistory(W0, No, Ni, P, sigma_o)
                       
sgd = SGD(lr=settings.lr)
model.compile(loss='mse',
              optimizer=sgd)

if settings.verbose:
    v = 1
else:
    v = 0
history = model.fit(X_train, Y_train,
                    batch_size=batch_sz, nb_epoch=nb_epoch,
                    verbose=v, validation_data=(X_test, Y_test), callbacks=[genhist])



if settings.savefile:
    np.savez(settings.savefile, train=np.asarray(history.history['loss']), test=np.asarray(history.history['val_loss']), exact_test=np.asarray(genhist.genhist), params=[settings])


if settings.showplot or settings.saveplot:
    epoch = np.linspace(0, nb_epoch, nb_epoch)

    fig, ax = plt.subplots(1)
    line1, = ax.plot(epoch, history.history['loss'], linewidth=2,label='Train loss')
    line2, = ax.plot(epoch, history.history['val_loss'], linewidth=2, label='Test loss')
    line3, = ax.plot(epoch, genhist.genhist, linewidth=2, label='Exact Test loss')
    ax.legend(loc='upper center')

    if settings.showplot:
        plt.show()
    elif settings.saveplot:
        fig.savefig('training_dynamics.pdf', bbox_inches='tight')


