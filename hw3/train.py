#!/usr/bin/env python
# -- coding: utf-8 --
"""
Convolutional Neural Network for Image Sentiment Classification
Training part
"""

import os
import argparse
import model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from utils import read_dataset, dump_history

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def main():
    """ Main function """
    # Handle options
    parser = argparse.ArgumentParser(prog='train.py', description='hw3 training script.')
    parser.add_argument('--input', type=str, metavar='<input_file>', default='./data/train_3.csv')
    parser.add_argument('--model', type=str, default='easy',
                        choices=['easy', 'simple', 'strong', 'dnn'], metavar='<model>')
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

    # Read data
    tra_feats, tra_labels = read_dataset(args.input)
    val_feats = read_dataset('./data/valid_3.csv', False)
    val_labels = read_dataset('./data/valid_3.csv', True, True)
    # tra_feats, tra_labels = read_dataset(args.input)
    # val_feats = read_dataset('./data/test.csv', False)
    # val_labels = read_dataset('./data/test_ans.csv', True, True)

    # Specify callbacks
    emotion_classifier = model.build_model(args.model, tra_labels.shape[1])
    history = model.History()
    model_name = os.path.join(store_path, "{epoch:02d}_model.hdf5")
    checkpoint = ModelCheckpoint(model_name, monitor='val_acc', verbose=0,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint, history]

    # Fit model
    if args.dataGen:
        print('Using ImageDataGenerator')

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True)

        datagen.fit(tra_feats)

        emotion_classifier.fit_generator(datagen.flow(tra_feats, tra_labels, batch_size=args.batch),
                                         steps_per_epoch=500,
                                         epochs=args.epoch,
                                         validation_data=(val_feats, val_labels),
                                         callbacks=callbacks_list)
    else:
        emotion_classifier.fit(tra_feats, tra_labels,
                               batch_size=args.batch,
                               epochs=args.epoch,
                               validation_data=(val_feats, val_labels),
                               callbacks=callbacks_list)

    # Save history and final model
    dump_history(store_path, history)
    emotion_classifier.save(os.path.join(store_path, 'final_model.hdf5'))

if __name__ == "__main__":

    main()
