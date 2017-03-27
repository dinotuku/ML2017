"""
Probability Generative Model for Binary Classification
For testing
"""
# -*- coding: utf8 -*-

import sys
import os
import math
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import pandas as pd

np.random.seed(2)

def read_test_data(x_name):
    """ Read testing data """
    print('Reading testing data')
    
    x_pandf = pd.read_csv(x_name, header=None, skiprows=1)

    return x_pandf

def process_test_data(x_df):
    """ Turn pandas dataframe into numpy array """
    x_data = np.array(x_df)

    return x_data

def pdf(data, mean, cov):
    """ Calculate Gaussian PDF"""
    prob = math.exp(-np.dot(np.dot((data - mean).transpose(), inv(cov)), (data - mean)) / 2) / \
           math.pow(math.pi * 2, mean.shape[0] / 2) / math.sqrt(math.fabs(det(cov)))

    return prob

def use_saved_model(x_data):
    """ Use model to classify """
    mod = np.load('./model/generative_model.npy')
    p_mean = mod[0]
    n_mean = mod[1]
    cov = mod[2]
    p_per = mod[3]
    n_per = mod[4]
    res = np.array([])

    print('Using model')
    for i in range(x_data.shape[0]):

        if pdf(x_data[i], p_mean, cov) != 0:

            pos_prob = pdf(x_data[i], p_mean, cov) * p_per / \
                       (pdf(x_data[i], p_mean, cov) * p_per + pdf(x_data[i], n_mean, cov) * n_per)

        else:

            pos_prob = 1.0

        if pos_prob > 0.5:

            res = np.append(res, 1)

        else:

            res = np.append(res, 0)

    return res

def save_result(res, name):
    """ Save testing result """
    print('Saving result')
    file = open(name, 'w')
    file.write('id,label\n')

    for i in range(len(res)):

        file.write('%d,%d\n' % (i + 1, res[i]))

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
