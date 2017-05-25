#!/usr/bin/env python
# -- coding: utf-8 --
"""
Multi-class & Multi-label Article Classification
Testing Part
"""

import os
import pickle
from argparse import ArgumentParser
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
# from keras.utils.vis_utils import plot_model
from utils import read_dataset
from train import f1_score

THRESHOLD = 0.5

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def main():
    """ Main function """
    parser = ArgumentParser()
    parser.add_argument('--bag', action='store_true', help='Use Bag-of-words model')
    parser.add_argument('--input', type=str, default='data/test_data.csv', help='Input file path')
    parser.add_argument('--output', type=str, default='res.csv', help='Output file path')
    parser.add_argument('--model', type=str, default='best', help='Use which model')
    args = parser.parse_args()

    test_text = read_dataset(args.input, 'test')
    tok_file = open(os.path.join(MODEL_DIR, 'bag_tokenizer.pkl')
                    if args.bag else os.path.join(MODEL_DIR, 'tokenizer.pkl'), 'rb')
    tokenizer = pickle.load(tok_file)
    tok_file.close()
    test_sequences = tokenizer.texts_to_sequences(test_text)
    if args.bag:
        test_data = tokenizer.sequences_to_matrix(test_sequences, mode='freq')
    else:
        test_data = pad_sequences(test_sequences)

    tagsidx = np.load(os.path.join(MODEL_DIR, 'tagsIdx.npy'))
    model = load_model(os.path.join(MODEL_DIR, "{:s}_bag_model.hdf5".format(args.model)
                                    if args.bag else "{:s}_model.hdf5".format(args.model)),
                       custom_objects={'f1_score': f1_score})
    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)

    res = model.predict(test_data, batch_size=64)
    for idx, item in enumerate(res):
        res[idx][item.argmax()] = 1
    res[res >= THRESHOLD] = 1
    res[res < THRESHOLD] = 0

    file = open(args.output, 'w')
    file.write('\"id\",\"tags\"\n')
    for idx, item in enumerate(res):
        ans = tagsidx[item == 1]
        file.write("\"{:d}\",\"{:s}\"\n".format(idx, ' '.join(ans)))

if __name__ == '__main__':

    main()
