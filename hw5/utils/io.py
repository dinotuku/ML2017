#!/usr/bin/env python
# -- coding: utf-8 --
"""
Functions for I/O
"""

import os
import numpy as np
from sklearn import preprocessing
from keras.utils import to_categorical

def read_dataset(path, mode):
    """ Read dataset for training and testing """
    if mode == 'train':
        tags = []
    texts = []
    file = open(path, 'r')
    lines = file.readlines()
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        line = line.split(',')
        if mode == 'train':
            tags.append(line[1][1:-1].split(' '))
            texts.append(''.join(line[2:]))
        else:
            texts.append(''.join(line[1:]))

    file.close()
    if mode == 'train':
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit([tag for sublist in tags for tag in sublist])
        if not os.path.isfile('./model/tagsIdx'):
            np.save('./model/tagsIdx', label_encoder.classes_)
        for idx, item in enumerate(tags):
            tags[idx] = label_encoder.transform(item)
            tags[idx] = to_categorical(tags[idx], num_classes=len(label_encoder.classes_))
            tags[idx] = tags[idx].sum(axis=0)

        tags = np.array(tags)

    texts = np.array(texts)
    if mode == 'train':
        return tags, texts

    return texts


def dump_history(store_path, logs):
    """ Dump training history """

    with open(os.path.join(store_path, 'train_loss'), 'a') as file:
        for loss in logs.tra_loss:
            file.write('{}\n'.format(loss))

    with open(os.path.join(store_path, 'train_f1score'), 'a') as file:
        for f1_score in logs.tra_f1:
            file.write('{}\n'.format(f1_score))

    with open(os.path.join(store_path, 'valid_loss'), 'a') as file:
        for loss in logs.val_loss:
            file.write('{}\n'.format(loss))

    with open(os.path.join(store_path, 'valid_f1score'), 'a') as file:
        for f1_score in logs.val_f1:
            file.write('{}\n'.format(f1_score))
