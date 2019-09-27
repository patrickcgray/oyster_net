#coding=utf-8

try:
    from keras.applications.inception_v3 import InceptionV3
except:
    pass

try:
    from keras.preprocessing import image
except:
    pass

try:
    from keras.models import Model
except:
    pass

try:
    from keras.layers import Dense, GlobalAveragePooling2D
except:
    pass

try:
    from keras.callbacks import ModelCheckpoint
except:
    pass

try:
    from keras import backend as K
except:
    pass

try:
    from keras.layers import Input
except:
    pass

try:
    import tensorflow as tf
except:
    pass

try:
    import cv2
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    import csv
except:
    pass

try:
    import os
except:
    pass

try:
    import random
except:
    pass

try:
    import matplotlib.pyplot as plt
except:
    pass

try:
    from keras.applications.densenet import preprocess_input
except:
    pass

try:
    import imgaug as ia
except:
    pass

try:
    from imgaug import augmenters as iaa
except:
    pass

try:
    from keras.optimizers import SGD
except:
    pass

try:
    from keras.applications.mobilenet_v2 import MobileNetV2
except:
    pass

try:
    from keras.optimizers import SGD
except:
    pass

try:
    from keras.optimizers import SGD
except:
    pass

try:
    from keras.applications.densenet import DenseNet121
except:
    pass

try:
    from keras.optimizers import SGD
except:
    pass

try:
    from keras.applications.densenet import DenseNet121
except:
    pass

try:
    from keras.applications.densenet import DenseNet121
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass

from __future__ import print_function

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.layers.core import Dense, Dropout, Activation
except:
    pass

try:
    from keras.datasets import mnist
except:
    pass

try:
    from keras.utils import np_utils
except:
    pass

try:
    import keras
except:
    pass

try:
    from keras.datasets import cifar10
except:
    pass

try:
    from keras.preprocessing.image import ImageDataGenerator
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.layers import Dense, Dropout, Activation, Flatten
except:
    pass

try:
    from keras.layers import Conv2D, MaxPooling2D
except:
    pass

try:
    import os
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    import matplotlib.pyplot as plt
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas.distributions import conditional

'''
Data providing function:
This function is separated from model() so that hyperopt
won't reload data for each evaluation run.
'''
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
nb_classes = 10
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


def keras_fmin_fnct(space):

    '''
    Model providing function:
    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(space['Dropout']))
    model.add(Dense(space['Dense']))
    model.add(Activation(space['Activation']))
    model.add(Dropout(space['Dropout_1']))

    # If we choose 'four', add an additional fourth layer
    if space['Dropout_2'] == 'four':
        model.add(Dense(100))
        model.add(space['add'])
        model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=space['optimizer'],
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=space['batch_size'],
              nb_epoch=1,
              verbose=2,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'Dropout': hp.uniform('Dropout', 0, 1),
        'Dense': hp.choice('Dense', [256, 512, 1024]),
        'Activation': hp.choice('Activation', ['relu', 'sigmoid']),
        'Dropout_1': hp.uniform('Dropout_1', 0, 1),
        'Dropout_2': hp.choice('Dropout_2', ['three', 'four']),
        'add': hp.choice('add', [Dropout(0.5), Activation('linear')]),
        'optimizer': hp.choice('optimizer', ['rmsprop', 'adam', 'sgd']),
        'batch_size': hp.choice('batch_size', [64, 128]),
    }
