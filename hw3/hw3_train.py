#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
Convolutional Neural Network for Image Sentiment Classification
Training part
"""

import sys
import os
import errno
import csv
import numpy  as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, save_model

GENERATORON = True
LOADMODEL = False

def read_train_data(name):
    """ Read training data """
    print('Reading training data')

    label = []
    feature = []
    train_file = open(name, 'r')

    for row in csv.DictReader(train_file):

        feature.append(row['feature'].split())
        label.append(row['label'])

    train_file.close()

    return feature, label

def process_train_data(x_list, y_list):
    """ Turn python list into numpy array and some preprocessing """
    print('Processing training data')

    x_data = np.array(x_list).astype('float32')
    x_data = x_data / 255
    x_data = x_data.reshape(x_data.shape[0], 48, 48, 1)
    y_data = np.array(y_list).astype(int)
    y_data = np_utils.to_categorical(y_data, 7)

    return x_data, y_data

def convolution_neural_network(tra_x, tra_y):
    """ Construct and evaluate a CNN """

    print('Constructing CNN model')

    feature = []
    test_file = open('./data/test.csv', 'r')

    for row in csv.DictReader(test_file):

        feature.append(row['feature'].split())

    test_file.close()

    x_data = np.array(feature).astype('float32')
    x_data = x_data / 255
    x_data = x_data.reshape(x_data.shape[0], 48, 48, 1)

    ans_df = []
    ans_file = open('./data/test_ans.csv', 'r')

    for row in csv.DictReader(ans_file):

        ans_df.append(row['label'])

    ans_file.close()
    ans_data = np.array(ans_df).astype(int)
    ans_data = np_utils.to_categorical(ans_data, 7)

    if LOADMODEL:

        model = load_model('./model/' + sys.argv[2] + '_model.hdf5')

    else:

        model = Sequential()

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(units=2048, activation='relu'))

        model.add(Dropout(0.7))
        model.add(Dense(units=2048, activation='relu'))

        model.add(Dropout(0.7))
        model.add(Dense(units=7, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta', metrics=['accuracy'])

    # checkpoint
    name = './model/{epoch:02d}_model.hdf5'

    if not os.path.exists(os.path.dirname(name)):

        try:
            os.makedirs(os.path.dirname(name))

        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    checkpoint = ModelCheckpoint(name, monitor='val_acc', verbose=0, \
                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print('Training CNN model')

    if GENERATORON:

        print('Using ImageDataGenerator')

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True)

        datagen.fit(tra_x)

        model.fit_generator(datagen.flow(tra_x, tra_y, batch_size=256),
                            steps_per_epoch=500, \
                            epochs=200, \
                            validation_data=(x_data, ans_data), \
                            callbacks=callbacks_list)

    else:

        model.fit(tra_x, tra_y, \
                  batch_size=256, \
                  epochs=100, \
                  validation_data=(x_data, ans_data), \
                  callbacks=callbacks_list)

    print('Saving final model')

    save_model(model, './model/final_model.hdf5')

def main():
    """ Main function """
    train_data_name = sys.argv[1]

    train_x_df, train_y_df = read_train_data(train_data_name)
    train_x_data, train_y_data = process_train_data(train_x_df, train_y_df)
    convolution_neural_network(train_x_data, train_y_data)

if __name__ == '__main__':

    main()
