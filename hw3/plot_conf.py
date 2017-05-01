#!/usr/bin/env python
# -- coding: utf-8 --
"""
Plot confusion matrix
"""

import os
import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils import read_dataset, loss

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def plot_confusion_matrix(cma, classes, title='Confusion matrix', cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cma = cma.astype('float') / cma.sum(axis=1)[:, np.newaxis]
    plt.imshow(cma, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cma.max() / 2.
    for i, j in itertools.product(range(cma.shape[0]), range(cma.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cma[i, j]), horizontalalignment="center",
                 color="white" if cma[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    """ Main function """
    parser = argparse.ArgumentParser(prog='plot_model.py', description='Plot the model.')
    parser.add_argument('--model', type=str, default='simple',
                        choices=['simple', 'easy', 'strong', 'dnn', 'semi'], metavar='<model>')
    parser.add_argument('--epoch', type=int, default=100, metavar='<#epoch>')
    parser.add_argument('--dataGen', type=bool, default=False, metavar='<use_dataGen(bool)>')
    parser.add_argument('--batch', type=int, default=256, metavar='<batch_size>')
    parser.add_argument('--idx', type=int, required=True, metavar='<suffix>')
    args = parser.parse_args()
    store_path = "{}_epoch{}{}_{}".format(args.model, args.epoch,
                                          ('_G' if args.dataGen else ''), args.idx)
    print("Loading model from {}".format(store_path))
    model_path = os.path.join(MODEL_DIR, store_path, 'best_model.hdf5')

    if (args.model == 'semi'):
        emotion_classifier = load_model(model_path, custom_objects={'loss': loss})
    else:
        emotion_classifier = load_model(model_path)

    np.set_printoptions(precision=2)
    val_feats = read_dataset('./data/valid_3.csv', False)
    predictions = emotion_classifier.predict_classes(val_feats, batch_size=args.batch)
    val_labels = read_dataset('./data/valid_3.csv', True, True)
    val_labels = argmax(val_labels, axis=1)
    conf_mat = confusion_matrix(val_labels, predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
    plt.show()

if __name__ == "__main__":

    main()
