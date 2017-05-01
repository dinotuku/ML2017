#!/usr/bin/env python
# -- coding: utf-8 --
"""
Functions for I/O
"""

import os
import csv
import numpy as np
from keras.utils.np_utils import to_categorical

def read_dataset(path, has_label=True, ans=False):
    """
    Return:

    features: (num_data, 48, 48, 1) float32 nparray

    labels: (num_data, num_class) float64 nparray

    """
    if has_label and not ans:
        labels = []
        features = []
    elif ans:
        labels = []
    else:
        features = []

    read_file = open(path, 'r')

    for line in csv.DictReader(read_file):
        if has_label:
            labels.append(line['label'])
        if not ans:
            features.append(line['feature'].split())

    read_file.close()

    if has_label:
        labels = to_categorical(np.array(labels, dtype='int32'))

    if not ans:
        features = np.array(features, dtype='float32')
        features = features / 255
        features = features.reshape(features.shape[0], 48, 48, 1)

    if has_label and not ans:
        return features, labels
    elif ans:
        return labels
    else:
        return features


def dump_history(store_path, logs):
    """ Dump training history """

    with open(os.path.join(store_path, 'train_loss'), 'a') as file:
        for loss in logs.tr_losses:
            file.write('{}\n'.format(loss))

    with open(os.path.join(store_path, 'train_accuracy'), 'a') as file:
        for acc in logs.tr_accs:
            file.write('{}\n'.format(acc))

    with open(os.path.join(store_path, 'valid_loss'), 'a') as file:
        for loss in logs.val_losses:
            file.write('{}\n'.format(loss))

    with open(os.path.join(store_path, 'valid_accuracy'), 'a') as file:
        for acc in logs.val_accs:
            file.write('{}\n'.format(acc))
