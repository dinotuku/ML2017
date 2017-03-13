# -*- coding: utf8 -*-
import sys
import numpy as np
import pandas as pd

train_data_name = sys.argv[1]
test_data_name = sys.argv[2]
output_name = sys.argv[3]

pd.options.mode.chained_assignment = None


def read_training_data(name):

  print('Reading training data')
  df = pd.read_csv(name, encoding='Big5', na_values='NR', header=None, skiprows=1)
  # replace NR by 0
  df.fillna(value=0, inplace=True)
  # drop 測站名
  df.drop(1, axis=1, inplace=True)

  print('Data is read: ')
  print(df.head())

  data = {}
  # group data in date
  for name, ddf in df.groupby(0):

      date = [s.zfill(2) for s in name.split('/')]
      month = date[1]

      # drop date column in ddf
      ddf.drop(0, axis=1, inplace=True)

      # set parameter into row index
      ddf.set_index(2, drop=True, inplace=True)

      # set date(m-d-h) into column index
      ddf.columns = ['-'.join(date[1:]+[str(i).zfill(2)]) for i in range(24)]
      if month in data:
        data[month] = pd.concat([data[month], ddf], axis=1)
      else:
        data[month] = ddf

  for key in data.keys():

    data[key] = data[key][data[key].columns.sort_values()]

  return data


def process_training_data(data):

  train_x = []
  train_y = []
  for i in range(1, 13):

    X = data[str(i).zfill(2)].values

    for j in range(len(X[0]) - 9):

      tmp_x = []

      for k in [9]:

        tmp_x.append(X[k][j:j + 8])

      train_x.append(tmp_x)

      # PM2.5 value
      train_y.append(X[9][j + 9])

  train_x = np.array(train_x)
  train_y = np.array(train_y)

  # add sqaured data
  # train_x_2_3 = np.append(train_x ** 2, train_x ** 3, axis=1)
  train_x = np.append(train_x, train_x ** 2, axis=1)

  return train_x, train_y


def cal_rmse(x, w, b, y):

  return np.sqrt(((y - b - (x*w).sum(axis=1).sum(axis=1)) ** 2).sum() / len(y))


def linear_regression(x, y):

  fNum = 2
  lr = 0.001
  batch = 50
  print_every = 100
  iteration = 2000

  w = np.random.rand(fNum, 8) / y.shape[0]
  b = np.average(y)

  b_lr = 0.0
  w_lr = np.zeros((fNum, 8))

  b_history = np.array(b)
  w_history = np.array(w)

  for i in range(iteration + 1):

    b_grad = 0.0
    w_grad = np.zeros((fNum, 8))

    train_rmse = []

    batchNum = 0
    diff = 0

    for n in np.random.permutation(x.shape[0]):

      if batchNum < batch:

        batchNum += 1
        b_grad = b_grad - 2.0*(y[n] - b - (w*x[n]).sum())
        w_grad = w_grad - 2.0*(y[n] - b - (w*x[n]).sum())*x[n]

      else:

        break
        
    b_lr += b_grad**2
    w_lr += w_grad**2
    b = b - lr/np.sqrt(b_lr / (i+1)) * b_grad
    w = w - lr/np.sqrt(w_lr / (i+1)) * w_grad
    b_history = np.append(b_history, b)
    w_history = np.append(w_history, w)

    rmse = cal_rmse(x, w, b, y)
    train_rmse.append(rmse)

    if i % print_every == 0:

      print('Iteration: %d, RMSE: %.4f' % (i, rmse))

  return w, b


def main():

  train_data = read_training_data(train_data_name)
  train_x, train_y = process_training_data(train_data)
  weight, bias = linear_regression(train_x, train_y)


if __name__ == '__main__':

  main()














