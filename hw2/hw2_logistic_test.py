"""
Logistic Regression Model for Binary Classification
Testing part
"""
# -*- coding: utf8 -*-

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

def powering(mat, power):
    """ Add powered feature """
    tmp = mat

    for i in range(2, power + 1):

        ptmp = tmp ** i
        index = [0, 1, 3, 4, 5]
        mat = np.append(mat, ptmp[:, index], axis=1)

    return mat

def sigmoid(arg):
    """ Calculate sigmoid value """
    res = 1 / (1.0 + np.exp(-arg))
    return np.clip(res, 0.00000000000001, 0.99999999999999)

def read_test_data(x_name):
    """ Read testing data """
    print('Reading testing data')

    x_pandf = pd.read_csv(x_name, header=None, skiprows=1)

    return x_pandf

def process_test_data(x_df, mod):
    """ Turn pandas dataframe into numpy array """
    power, del_list = mod[4], mod[5]
    x_data = np.array(x_df).astype(float)
    x_data = np.delete(x_data, del_list, 1)
    x_data = powering(x_data, power)

    return x_data

def use_saved_model(x_data, mod):
    """ Use model to classify """
    weight, bias = mod[0], mod[1]
    mean, devi = mod[2], mod[3]
    res = np.array([])

    print('Using model')

    for i in range(x_data.shape[0]):

        x_data[i] = (x_data[i] - mean) / devi

        prob = sigmoid(np.dot(x_data[i], weight) + bias)

        if prob > 0.5:

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
    model_name = sys.argv[3]

    model = np.load('./model/logistic_model_' + model_name + '.npy')
    test_x_df = read_test_data(test_x_data_name)
    test_x_data = process_test_data(test_x_df, model)

    result = use_saved_model(test_x_data, model)
    save_result(result, output_name)

if __name__ == '__main__':

    main()
