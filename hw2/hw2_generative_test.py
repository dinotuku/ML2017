"""
Probability Generative Model for Binary Classification
Testing part
"""
# -*- coding: utf8 -*-

import sys
import numpy as np
from numpy.linalg import inv
import pandas as pd

np.random.seed(2)

def read_test_data(x_name):
    """ Read testing data """
    print('Reading testing data')

    x_pandf = pd.read_csv(x_name, header=None, skiprows=1)

    return x_pandf

def process_test_data(x_df):
    """ Turn pandas dataframe into numpy array """
    x_data = np.array(x_df).astype(float)

    return x_data

def use_saved_model(x_data):
    """ Use model to classify """
    mod = np.load('./model/generative_model.npy')
    p_mean, n_mean, cov = mod[0], mod[1], mod[2]
    p_per, n_per = mod[3], mod[4]
    mean, devi = mod[5], mod[6]
    res = np.array([])

    print('Using model')
    for i in range(x_data.shape[0]):

        x_data[i] = (x_data[i] - mean) / devi

        pos_tmp = -np.dot(np.dot((x_data[i] - p_mean).transpose(), inv(cov)), (x_data[i] - p_mean))
        neg_tmp = -np.dot(np.dot((x_data[i] - n_mean).transpose(), inv(cov)), (x_data[i] - n_mean))
        pos_prob = np.exp(pos_tmp * 0.5) * p_per
        neg_prob = np.exp(neg_tmp * 0.5) * n_per

        if pos_prob > neg_prob:

            res = np.append(res, 1)

        else:

            res = np.append(res, 0)

    return res

def save_result(res, name):
    """ Save testing result """
    print('Saving result')
    out = open(name, 'w')
    out.write('id,label\n')

    for ind, val in enumerate(res):

        out.write('%d,%d\n' % (ind + 1, val))

def main():
    """ Main function """
    test_x_data_name = sys.argv[1]
    output_name = sys.argv[2]

    test_x_df = read_test_data(test_x_data_name)
    test_x_data = process_test_data(test_x_df)
    result = use_saved_model(test_x_data)
    save_result(result, output_name)

if __name__ == '__main__':

    main()
