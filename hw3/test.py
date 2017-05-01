#!/usr/bin/env python
# -- coding: utf-8 --
"""
Convolutional Neural Network for Image Sentiment Classification
Testing part
"""

import os
import argparse
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from scipy.stats import entropy
import numpy as np
from numpy import argmax
from utils import read_dataset, loss

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def main():
    """ Main function """
    parser = argparse.ArgumentParser(prog='predict.py',
                                     description='ML-Assignment3 testing script.')
    parser.add_argument('--input', type=str, metavar='<input_file>', default='./data/test.csv')
    parser.add_argument('--output', type=str, metavar='<output_file>', default='prediction.csv')
    parser.add_argument('--model', type=str, default='easy',
                        choices=['easy', 'simple', 'strong', 'dnn', 'semi'], metavar='<model>')
    parser.add_argument('--epoch', type=int, default=100, metavar='<#epoch>')
    parser.add_argument('--dataGen', type=bool, default=False, metavar='<use_dataGen(bool)>')
    parser.add_argument('--batch', type=int, default=256, metavar='<batch_size>')
    parser.add_argument('--idx', type=int, required=True, metavar='<suffix>')
    parser.add_argument('--choice', type=str, required=True, metavar='<model_choice>')
    args = parser.parse_args()

    store_path = "{}_epoch{}{}_{}".format(args.model, args.epoch,
                                          ('_G' if args.dataGen else ''), args.idx)
    print("Loading model from {}".format(store_path))
    model_path = os.path.join(MODEL_DIR, store_path, '{}_model.hdf5'.format(args.choice))

    if (args.model == 'semi'):
        emotion_classifier = load_model(model_path, custom_objects={'loss': loss})
    else:
        emotion_classifier = load_model(model_path)

    emotion_classifier.summary()
    test_feats = read_dataset(args.input, False)

    ans = emotion_classifier.predict_classes(test_feats, batch_size=args.batch)
    with open(args.output, 'w') as write_file:
        write_file.write("id,label\n")
        for idx, label in enumerate(ans):
            write_file.write("{},{}\n".format(idx, label))

    ans_data = read_dataset('./data/test_ans.csv', True, True)
    ans_data = argmax(ans_data, axis=1)

    tmp = 0

    for idx, label in enumerate(ans):
        if label == ans_data[idx]:
            tmp += 1

    acc = float(tmp / ans_data.shape[0])
    print('All test accuracy:', acc)

if __name__ == "__main__":

    main()
