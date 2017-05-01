#!/usr/bin/env python
# -- coding: utf-8 --
"""
Convolutional Neural Network for Image Sentiment Classification
Semi-supervised
Training part
"""

import os
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from scipy.stats import entropy
import numpy as np
import model
from utils import read_dataset, dump_history

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def cal_entropy(arr):
    """ Calculate entropy """
    entro = np.array([])
    for idx in range(arr.shape[0]):
        entro = np.append(entro, entropy(arr[idx]))
    return entro

def entropy_based_losses(unlab_y_pred):
    """ Custom losses function """
    entro = cal_entropy(unlab_y_pred)
    entro = np.sum(entro)
    def loss(y_true, y_pred):
        """ Keras losses function """
        cross = categorical_crossentropy(y_true, y_pred)
        return 0.9 * cross + 0.1 * entro
    return loss

def main():
    """ Main function """
    # Handle options
    parser = argparse.ArgumentParser(prog='train.py', description='hw3 training script.')
    parser.add_argument('--input', type=str, metavar='<input_file>', default='./data/train_3.csv')
    parser.add_argument('--model', type=str, default='semi',
                        choices=['semi'], metavar='<model>')
    parser.add_argument('--epoch', type=int, default=100, metavar='<#epoch>')
    parser.add_argument('--dataGen', type=bool, default=False, metavar='<use_dataGen(bool)>')
    parser.add_argument('--batch', type=int, default=256, metavar='<batch_size>')
    args = parser.parse_args()

    # Set saving path
    dir_cnt = 0
    log_path = "{}_epoch{}{}".format(args.model, str(args.epoch), ('_G' if args.dataGen else ''))
    log_path += '_'
    store_path = os.path.join(MODEL_DIR, log_path + str(dir_cnt))
    while True:
        if not os.path.isdir(store_path):
            os.makedirs(store_path)
            break
        else:
            dir_cnt += 1
            store_path = os.path.join(MODEL_DIR, log_path + str(dir_cnt))

    # Specify callbacks
    emotion_classifier = Sequential()

    # CNN part (you can repeat this part several times)
    emotion_classifier.add(Conv2D(64, 3, padding='same', activation='relu',
                           input_shape=(48, 48, 1)))
    emotion_classifier.add(Conv2D(64, 3, padding='same', activation='relu'))
    emotion_classifier.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_classifier.add(Dropout(0.5))

    emotion_classifier.add(Conv2D(128, 3, padding='same', activation='relu'))
    emotion_classifier.add(Conv2D(128, 3, padding='same', activation='relu'))
    emotion_classifier.add(Conv2D(128, 3, padding='same', activation='relu'))
    emotion_classifier.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_classifier.add(Dropout(0.5))

    emotion_classifier.add(Conv2D(256, 3, padding='same', activation='relu'))
    emotion_classifier.add(Conv2D(256, 3, padding='same', activation='relu'))
    emotion_classifier.add(Conv2D(256, 3, padding='same', activation='relu'))
    emotion_classifier.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_classifier.add(Dropout(0.5))

    # Fully connected part
    emotion_classifier.add(Flatten())

    emotion_classifier.add(Dense(2048, activation='relu'))

    emotion_classifier.add(Dropout(0.5))
    emotion_classifier.add(Dense(2048, activation='relu'))

    emotion_classifier.add(Dropout(0.5))
    emotion_classifier.add(Dense(7, activation='softmax'))

    opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    # emotion_classifier.compile(loss='categorical_crossentropy',
    #                            optimizer=opt,
    #                            metrics=['accuracy'])

    emotion_classifier.summary() # show the whole model in terminal

    unlabel_nb = 20000
    append_steps = 10
    final_steps = 100
    data_per_step = int(unlabel_nb / append_steps)

    # Read data
    # feats, labels = read_dataset(args.input)
    feats, labels = read_dataset('./data/train.csv')
    unlab_feats, unlab_labels = feats[:unlabel_nb], labels[:unlabel_nb]
    tra_feats, tra_labels = feats[unlabel_nb:], labels[unlabel_nb:]
    # val_feats = read_dataset('./data/valid_3.csv', False)
    # val_labels = read_dataset('./data/valid_3.csv', True, True)
    val_feats = read_dataset('./data/test.csv', False)
    val_labels = read_dataset('./data/test_ans.csv', True, True)

    # Specify callbacks function
    history = model.History()
    model_name = os.path.join(store_path, "{epoch:02d}_model.hdf5")
    checkpoint = ModelCheckpoint(model_name, monitor='val_acc', verbose=0,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint, history]

    # Fit model
    for _ in range(append_steps):
        unlab_res = emotion_classifier.predict(unlab_feats, batch_size=args.batch)

        emotion_classifier.compile(loss=entropy_based_losses(unlab_res),
                                   optimizer=opt,
                                   metrics=['accuracy'])
        if args.dataGen:
            print('Using ImageDataGenerator')

            datagen = ImageDataGenerator(
                rotation_range=25,
                width_shift_range=0.25,
                height_shift_range=0.25,
                zoom_range=0.25,
                horizontal_flip=True)

            datagen.fit(tra_feats)

            emotion_classifier.fit_generator(datagen.flow(tra_feats, tra_labels,
                                                          batch_size=args.batch),
                                             steps_per_epoch=300,
                                             epochs=(args.epoch - final_steps)/append_steps,
                                             validation_data=(val_feats, val_labels),
                                             callbacks=callbacks_list)
        else:
            emotion_classifier.fit(tra_feats, tra_labels,
                                   batch_size=args.batch,
                                   epochs=(args.epoch - final_steps)/append_steps,
                                   validation_data=(val_feats, val_labels),
                                   callbacks=callbacks_list)

        entro = cal_entropy(unlab_res)
        entro_sort_idx = np.argsort(entro)
        unlab_feats = unlab_feats[entro_sort_idx]
        unlab_labels = unlab_labels[entro_sort_idx]

        tra_feats = np.append(tra_feats,
                              unlab_feats[:data_per_step], axis=0)
        tra_labels = np.append(tra_labels,
                               unlab_labels[:data_per_step], axis=0)
        unlab_feats = np.delete(unlab_feats, np.s_[:data_per_step], 0)
        unlab_labels = np.delete(unlab_labels, np.s_[:data_per_step], 0)

    emotion_classifier.compile(loss='categorical_crossentropy',
                               optimizer=opt,
                               metrics=['accuracy'])

    if args.dataGen:
        print('Using ImageDataGenerator')

        datagen = ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.25,
            height_shift_range=0.25,
            zoom_range=0.25,
            horizontal_flip=True)

        datagen.fit(tra_feats)

        emotion_classifier.fit_generator(datagen.flow(tra_feats, tra_labels,
                                                      batch_size=args.batch),
                                         steps_per_epoch=300,
                                         epochs=final_steps,
                                         validation_data=(val_feats, val_labels),
                                         callbacks=callbacks_list)
    else:
        emotion_classifier.fit(tra_feats, tra_labels,
                               batch_size=args.batch,
                               epochs=final_steps,
                               validation_data=(val_feats, val_labels),
                               callbacks=callbacks_list)

    # Save history and final model
    dump_history(store_path, history)
    emotion_classifier.save(os.path.join(store_path, 'final_model.hdf5'))

if __name__ == "__main__":

    main()
