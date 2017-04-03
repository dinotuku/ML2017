"""
Probability Generative Model for Binary Classification
Training part
"""
# -*- coding: utf8 -*-

import errno
import os
import sys
import numpy as np
from numpy.linalg import inv
import pandas as pd

def normalize(mat):
    """ Normalize input data """
    mean = np.zeros(mat.shape[1])
    devi = np.ones(mat.shape[1])

    all_mean = np.mean(mat, axis=0)
    all_devi = np.std(mat, axis=0)

    index = np.array([0, 1, 3, 4, 5])
    mean[index] = all_mean[index]
    devi[index] = all_devi[index]

    mat_normed = (mat - mean) / devi

    return mat_normed, mean, devi

def read_train_data(x_name, y_name):
    """ Read training data """
    print('Reading training data')

    x_pandf = pd.read_csv(x_name, header=None, skiprows=1)
    y_pandf = pd.read_csv(y_name, header=None)

    return x_pandf, y_pandf

def process_train_data(x_df, y_df):
    """ Turn pandas dataframe into numpy array and seperate validation data """
    x_data = np.array(x_df).astype(float)
    y_data = np.array(y_df).astype(float)

    x_data, mean, devi = normalize(x_data)

    p_mean, n_mean, cov, p_per, n_per = get_mean_covariance(x_data, y_data)
    model = [p_mean, n_mean, cov, p_per, n_per, mean, devi]

    return x_data, y_data, model

def get_mean_covariance(x_data, y_data):
    """ Get the maximum likelihood mean and covariance """
    print('Get best mean and covariance')

    p_indices = np.where(y_data == 1)[0]
    n_indices = np.where(y_data == 0)[0]
    p_data = x_data[p_indices]
    n_data = x_data[n_indices]
    p_mean = np.mean(p_data, axis=0)
    n_mean = np.mean(n_data, axis=0)
    p_per = p_data.shape[0] / x_data.shape[0]
    n_per = n_data.shape[0] / x_data.shape[0]
    cov = p_per * np.cov(p_data.transpose()) + n_per * np.cov(n_data.transpose())

    return p_mean, n_mean, cov, p_per, n_per

def use_model(x_data, y_data, mod):
    """ Use model to validate classification result """
    p_mean, n_mean, cov = mod[0], mod[1], mod[2]
    p_per, n_per = mod[3], mod[4]
    res = np.array([])

    print('Validating')

    for i in range(x_data.shape[0]):

        pos_tmp = -np.dot(np.dot((x_data[i] - p_mean).transpose(), inv(cov)), (x_data[i] - p_mean))
        neg_tmp = -np.dot(np.dot((x_data[i] - n_mean).transpose(), inv(cov)), (x_data[i] - n_mean))
        pos_prob = np.exp(pos_tmp * 0.5) * p_per
        neg_prob = np.exp(neg_tmp * 0.5) * n_per

        if pos_prob > neg_prob:

            res = np.append(res, 1)

        else:

            res = np.append(res, 0)

    acc = np.where((res == y_data[0]) == 1)[0].shape[0] / y_data.shape[0]

    print('Training accuracy: ' + str(acc))

def save_model(mod):
    """ Save model for future testing """
    name = './model/generative_model'

    if not os.path.exists(os.path.dirname(name)):

        try:
            os.makedirs(os.path.dirname(name))

        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    print('Saving model')
    np.save(name, mod)

def main():
    """ Main function """
    train_x_data_name = sys.argv[1]
    train_y_data_name = sys.argv[2]

    train_x_df, train_y_df = read_train_data(train_x_data_name, train_y_data_name)
    train_x_data, train_y_data, model = process_train_data(train_x_df, train_y_df)
    use_model(train_x_data, train_y_data, model)
    save_model(model)

if __name__ == '__main__':

    main()
