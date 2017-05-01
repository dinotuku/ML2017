#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
Convolutional Neural Network for Image Sentiment Classification
Testing part
"""

import sys
import csv
import numpy  as np
from keras.models import load_model

def read_test_data(name):
    """ Read testing data """
    print('Reading testing data')

    feature = []
    test_file = open(name, 'r')

    for row in csv.DictReader(test_file):

        feature.append(row['feature'].split())

    test_file.close()

    return feature

def process_test_data(x_df):
    """ Turn pandas dataframe into numpy array and seperate validation data """
    print('Processing testing data')

    x_data = np.array(x_df).astype('float32')
    x_data = x_data / 255
    x_data = x_data.reshape(x_data.shape[0], 48, 48, 1)

    return x_data

def use_saved_model(x_data, name):
    """ Use model to classify """
    print('Loading CNN model')
    model = load_model('./model/strong_epoch0_G_0/' + sys.argv[3] + '_model.hdf5')
    result = model.predict(x_data, batch_size=128, verbose=1)
    result = result.argmax(1)

    print('Saving result')
    out = open(name, 'w')
    out.write('id,label\n')

    for ind, val in enumerate(result):

        out.write('%d,%d\n' % (ind, val))

    return result


def main():
    """ Main function """
    test_data_name = sys.argv[1]
    output_name = sys.argv[2]

    test_x_df = read_test_data(test_data_name)
    test_x_data = process_test_data(test_x_df)
    result = use_saved_model(test_x_data, output_name)

    ans_df = []
    ans_file = open('./data/test_ans.csv', 'r')

    for row in csv.DictReader(ans_file):

        ans_df.append(row['label'])

    ans_file.close()
    ans_data = np.array(ans_df).astype(int)

    tmp = 0

    for ind, val in enumerate(ans_data):

        if val == result[ind]:

            tmp += 1

    acc = float(tmp / ans_data.shape[0])
    print('All test accuracy:', acc)

if __name__ == '__main__':

    main()
