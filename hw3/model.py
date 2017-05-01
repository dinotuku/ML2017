#!/usr/bin/env python
# -- coding: utf-8 --
"""
Functions for building model and saving history
"""

import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.callbacks import Callback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class History(Callback):
    """ Class for training history """
    def on_train_begin(self, logs=None):
        """ Initialization """
        self.tr_losses = []
        self.val_losses = []
        self.tr_accs = []
        self.val_accs = []

    def on_epoch_end(self, epoch, logs=None):
        """ Log training information """
        logs = logs or {}
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))

def build_model(mode, nb_class):
    """
    Return the Keras model for training

    Keyword arguments:
    mode: model name specified in training and predicting script
    nb_class: number of classes in the problem

    """
    model = Sequential()

    if mode == 'easy':
        # CNN part (you can repeat this part several times)
        model.add(Conv2D(64, 3, padding='valid', activation='relu', input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, 3, padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, 3, padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        # Fully connected part
        model.add(Flatten())

        model.add(Dense(128, activation='relu'))

        model.add(Dropout(0.5))
        model.add(Dense(nb_class, activation='softmax'))

        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    elif mode == 'simple':
        # CNN part (you can repeat this part several times)
        model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=(48, 48, 1)))
        model.add(Conv2D(64, 3, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(128, 3, padding='same', activation='relu'))
        model.add(Conv2D(128, 3, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        # Fully connected part
        model.add(Flatten())

        model.add(Dense(256, activation='relu'))

        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))

        model.add(Dropout(0.5))
        model.add(Dense(nb_class, activation='softmax'))

        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    elif mode == 'strong':
        # CNN part (you can repeat this part several times)
        model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=(48, 48, 1)))
        model.add(Conv2D(64, 3, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(128, 3, padding='same', activation='relu'))
        model.add(Conv2D(128, 3, padding='same', activation='relu'))
        model.add(Conv2D(128, 3, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(256, 3, padding='same', activation='relu'))
        model.add(Conv2D(256, 3, padding='same', activation='relu'))
        model.add(Conv2D(256, 3, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        # Fully connected part
        model.add(Flatten())

        model.add(Dense(2048, activation='relu'))

        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))

        model.add(Dropout(0.5))
        model.add(Dense(nb_class, activation='softmax'))

        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    elif mode == 'dnn':
        model.add(Flatten(input_shape=(48, 48, 1)))

        model.add(Dense(2048, activation='relu'))

        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))

        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))

        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))

        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))

        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))

        model.add(Dropout(0.5))
        model.add(Dense(nb_class, activation='softmax'))

        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    elif mode == 'semi':
        # CNN part (you can repeat this part several times)
        model.add(Conv2D(64, 3, padding='valid', activation='relu', input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, 3, padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, 3, padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        # Fully connected part
        model.add(Flatten())

        model.add(Dense(128, activation='relu'))

        model.add(Dropout(0.5))
        model.add(Dense(nb_class, activation='softmax'))

        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

    model.summary() # show the whole model in terminal

    return model
