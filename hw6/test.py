#!/usr/bin/env python
# -- coding: utf-8 --
"""
Matrix Factorization For Movie Ratings Prediction
"""

import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from keras.models import load_model

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def main():
    """ Main function """
    parser = ArgumentParser()
    parser.add_argument('--input', default='data/', help='Input file path')
    parser.add_argument('--output', default='res.csv', help='Output file path')
    parser.add_argument('--model', default='best', help='Use which model to test')
    parser.add_argument('--dnn', action='store_true', help='Use DNN model')
    parser.add_argument('--normal', action='store_true', help='Normalize ratings')
    args = parser.parse_args()

    pairs_df = pd.read_csv(os.path.join(args.input, 'test.csv'), sep=',')
    print("{:d} user-movie pairs loaded".format(len(pairs_df)))

    users = pairs_df['UserID'].values - 1
    print("Users: {:s}, shape = {:s}".format(str(users), str(users.shape)))
    movies = pairs_df['MovieID'].values - 1
    print("Movies: {:s}, shape = {:s}".format(str(movies), str(movies.shape)))

    model = load_model(os.path.join(MODEL_DIR, "{:s}_dnn_model.hdf5".format(args.model)
                                    if args.dnn else "{:s}_mf_model.hdf5".format(args.model)))

    res = model.predict([users, movies])
    res = np.array(res).reshape(len(pairs_df))

    if args.normal:
        ratings_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'train.csv'),
                                 sep=',')
        ratings = ratings_df['Rating'].values
        res = (res * np.std(ratings)) + np.mean(ratings)

    res[res > 5] = 5
    res[res < 1] = 1

    file = open(args.output, 'w')
    file.write('TestDataID,Rating\n')
    for idx, item in enumerate(res):
        file.write("{:d},{:.4f}\n".format(idx + 1, item))

if __name__ == '__main__':

    main()
