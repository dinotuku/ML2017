"""
Logistic Regression Model for Binary Classification
Training part
"""
# -*- coding: utf8 -*-

import errno
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

np.random.seed(0)

def powering(mat, power):
    """ Add powered feature """
    tmp = mat

    for i in range(2, power + 1):

        ptmp = tmp ** i
        index = [0, 1, 3, 4, 5]
        mat = np.append(mat, ptmp[:, index], axis=1)

    return mat

def normalize(mat):
    """ Normalize input data """
    mean = np.zeros(mat.shape[1])
    devi = np.ones(mat.shape[1])

    all_mean = np.mean(mat, axis=0)
    all_devi = np.std(mat, axis=0)

    index = np.array([0, 1, 3, 4, 5, 105, 106, 107, 108, 109])
    mean[index] = all_mean[index]
    devi[index] = all_devi[index]

    mat_normed = (mat - mean) / devi

    return mat_normed, mean, devi

def sigmoid(arg):
    """ Calculate sigmoid value """
    res = 1 / (1.0 + np.exp(-arg))
    return np.clip(res, 0.00000000000001, 0.99999999999999)

def cal_accuracy(inp, outp, weight, bias, per):
    """ Calculate accuracy for classification result """
    res = np.array([])
    count = 0

    for i in range(inp.shape[0]):

        prob = sigmoid(np.dot(inp[i], weight) + bias)

        if prob > per:

            res = np.append(res, 1)

        else:

            res = np.append(res, 0)

        if res[i] == outp[i]:

            count += 1

    acc = count / outp.shape[0]

    return acc

def read_train_data(x_name, y_name):
    """ Read training data """
    print('Reading training data')

    x_pandf = pd.read_csv(x_name, header=None, skiprows=1)
    y_pandf = pd.read_csv(y_name, header=None)

    return x_pandf, y_pandf

def process_train_data(x_df, y_df, power, del_list):
    """ Turn pandas dataframe into numpy array and seperate validation data """
    x_data = np.array(x_df).astype(float)
    x_data = np.delete(x_data, del_list, 1)
    x_data = powering(x_data, power)
    y_data = np.array(y_df)

    indices = np.random.permutation(x_data.shape[0])
    train_size = int(x_data.shape[0] * 0.8)
    tra_id, val_id = indices[:train_size], indices[train_size:]
    tra_x, tra_y = x_data[tra_id], y_data[tra_id]
    tra_x_normed, mean, devi = normalize(tra_x)

    n_mod = [mean, devi]
    val_x, val_y = x_data[val_id], y_data[val_id]

    return tra_x_normed, tra_y, val_x, val_y, n_mod

def gradient_descent(tra_x, tra_y, val_x, val_y, mod):
    """ Gradient descent for logistic regression """
    # normailize validation data
    mean = mod[0]
    devi = mod[1]

    val_x_normed = (val_x - mean) / devi

    # set gradient descent parameters
    i_lr = 1.2
    batch = int(tra_x.shape[0] * 0.01)
    print_every = 10
    save_every = 10
    iteration = 1000

    weight = np.random.rand(tra_x.shape[1]) / tra_x.shape[0]
    bias = 0.0
    delta_b = 0.0
    delta_w = np.zeros(tra_x.shape[1])

    # Adagrad parameters
    b_lr = 0.0
    w_lr = np.zeros(tra_x.shape[1])
    epsilon = 1e-6

    # Adam parameters
    # beta1 = 0.9
    # beta2 = 0.999
    # epsilon = 1e-8
    # b_m = b_v = 0.0
    # w_m = w_v = np.zeros(tra_x.shape[1])

    # Regularization and Momentum parameters
    lamb = 0.0
    momen = 0.7

    print('Starting to train model')

    for i in range(iteration + 1):

        b_grad = 0.0
        w_grad = np.zeros(tra_x.shape[1])

        batch_num = 0
        # mini-batch training
        for j in np.random.permutation(tra_x.shape[0]):

            if batch_num < batch:

                batch_num += 1
                b_grad -= tra_y[j] - sigmoid(bias + np.dot(weight, tra_x[j]))
                w_grad -= (tra_y[j] - sigmoid(bias + np.dot(weight, tra_x[j]))) * tra_x[j]

            else:

                b_grad -= tra_y[j] - sigmoid(bias + np.dot(weight, tra_x[j]))
                w_grad -= (tra_y[j] - sigmoid(bias + np.dot(weight, tra_x[j]))) * tra_x[j]

                # Adagrad + Momentum + Regularization
                b_lr += b_grad ** 2
                w_lr += w_grad ** 2
                delta_b = momen * delta_b + (1 - momen) * i_lr / \
                          np.sqrt(b_lr + epsilon) * b_grad
                delta_w = momen * delta_w + (1 - momen) * i_lr / \
                          np.sqrt(w_lr + epsilon) * (w_grad + lamb * weight)
                bias -= delta_b
                weight -= delta_w

                # Adam
                # b_m = beta1 * b_m + (1 - beta1) * b_grad
                # b_v = beta2 * b_v + (1 - beta2) * b_grad ** 2
                # w_m = beta1 * w_m + (1 - beta1) * w_grad
                # w_v = beta2 * w_v + (1 - beta2) * w_grad ** 2
                # b_mt = b_m / (1 - beta1 ** (i + 1))
                # b_vt = b_v / (1 - beta2 ** (i + 1))
                # w_mt = w_m / (1 - beta1 ** (i + 1))
                # w_vt = w_v / (1 - beta2 ** (i + 1))
                # bias -= i_lr / (np.sqrt(b_vt) + epsilon) * b_mt
                # weight -= i_lr / (np.sqrt(w_vt) + epsilon) * (w_mt + lamb * weight)

                b_grad = 0.0
                w_grad = np.zeros(tra_x.shape[1])
                batch_num = 0

        tacc = cal_accuracy(tra_x, tra_y, weight, bias, 0.5)
        vacc = cal_accuracy(val_x_normed, val_y, weight, bias, 0.5)

        if i % print_every == 0:

            print('Iteration: %d, TACC: %.4f, VACC: %.4f' % (i, tacc, vacc))

        if i % save_every == 0:

            print('Saving model')
            power, del_list = mod[2], mod[3]
            model = [weight, bias, mean, devi, power, del_list]
            name = './model/logistic_model_'
            if not os.path.exists(os.path.dirname(name)):

                try:
                    os.makedirs(os.path.dirname(name))

                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise

            np.save(name + str(i), model)

def main():
    """ Main function """
    train_x_data_name = sys.argv[1]
    train_y_data_name = sys.argv[2]
    power = 2
    del_list = [78]

    train_x_df, train_y_df = read_train_data(train_x_data_name, train_y_data_name)
    train_x_data, train_y_data, val_x_data, val_y_data, \
        model = process_train_data(train_x_df, train_y_df, power, del_list)
    model.extend((power, del_list))
    gradient_descent(train_x_data, train_y_data, val_x_data, val_y_data, model)

if __name__ == '__main__':

    main()
