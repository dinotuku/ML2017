"""
Probability Generative Model for Binary Classification
For training
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

def read_train_data(x_name, y_name):
    """ Read training data """
    print('Reading training data')
    
    x_pandf = pd.read_csv(x_name, header=None, skiprows=1)
    y_pandf = pd.read_csv(y_name, header=None)

    return x_pandf, y_pandf

def process_train_data(x_df, y_df):
    """ Turn pandas dataframe into numpy array and seperate validation data """
    x_data = np.array(x_df)
    y_data = np.array(y_df)

    cov = np.array([[1, 0], [0, 1]])

    while isnt_singular(cov):
    
        indices = np.random.permutation(x_data.shape[0])
        train_size = int(x_data.shape[0] * 0.9)
        tra_id, val_id = indices[:train_size], indices[train_size:]
        tra_x, tra_y = x_data[tra_id], y_data[tra_id]
        val_x, val_y = x_data[val_id], y_data[val_id]

        p_mean, n_mean, cov, p_per, n_per = get_mean_covariance(tra_x, tra_y)

    return val_x, val_y, p_mean, n_mean, cov, p_per, n_per

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

def isnt_singular(mat):
    """ Check singular matrix """
    return mat.shape[0] == mat.shape[1] and np.linalg.matrix_rank(mat) == mat.shape[0]

def pdf(data, mean, cov):
    """ Calculate Gaussian PDF"""
    prob = math.exp(-np.dot(np.dot((data - mean).transpose(), inv(cov)), (data - mean)) / 2) / \
           math.pow(math.pi * 2, mean.shape[0] / 2) / math.sqrt(math.fabs(det(cov)))

    return prob

def use_model(v_x_data, v_y_data, p_mean, n_mean, cov, p_per, n_per):
    """ Use model to validate classification result """
    res = np.array([])

    print('Validating')
    for i in range(v_x_data.shape[0]):

        if pdf(v_x_data[i], p_mean, cov) != 0:

            pos_prob = pdf(v_x_data[i], p_mean, cov) * p_per / \
                (pdf(v_x_data[i], p_mean, cov) * p_per + pdf(v_x_data[i], n_mean, cov) * n_per)

        else:

            pos_prob = 1.0

        if pos_prob > 0.6:

            res = np.append(res, 1)

        else:

            res = np.append(res, 0)

    acc = np.where((res == v_y_data[0]) == 1)[0].shape[0] / v_y_data.shape[0]

    print('Validation accuracy: ' + str(acc))

def save_model(p_mean, n_mean, cov, p_per, n_per):
    """ Save model for future testing """
    name = './model/generative_model'

    if not os.path.exists(os.path.dirname(name)):

        try:
            os.makedirs(os.path.dirname(name))

        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    print('Saving model')
    model = [p_mean, n_mean, cov, p_per, n_per]
    np.save(name, model)

def main():
    """ Main function """
    train_x_data_name = sys.argv[1]
    train_y_data_name = sys.argv[2]

    train_x_df, train_y_df = read_train_data(train_x_data_name, train_y_data_name)
    val_x_data, val_y_data, pos_class_mean, neg_class_mean, \
        cov, pos_per, neg_per = process_train_data(train_x_df, train_y_df)
    use_model(val_x_data, val_y_data, pos_class_mean, neg_class_mean, \
        cov, pos_per, neg_per)
    save_model(pos_class_mean, neg_class_mean, cov, pos_per, neg_per)

if __name__ == '__main__':

    main()
