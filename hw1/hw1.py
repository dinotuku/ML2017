# -*- coding: utf8 -*-
import sys, os
import numpy as np
import pandas as pd

train_data_name = sys.argv[1]
test_data_name = sys.argv[2]
output_name = sys.argv[3]

feat_list = [5, 6, 9]

pd.options.mode.chained_assignment = None


def read_training_data(name):

  print('Reading training data')
  df = pd.read_csv(name, encoding='Big5', na_values='NR', header=None, skiprows=1)
  
  # replace NR by 0
  df.fillna(value=0, inplace=True)
  
  # drop 測站名
  df.drop(1, axis=1, inplace=True)

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

      for k in feat_list:

        tmp_x.append(X[k][j:j + 9])

      train_x.append(tmp_x)

      # PM2.5 value
      train_y.append(X[9][j + 9])

  train_x = np.array(train_x)
  train_y = np.array(train_y)

  # add sqaured data
  # train_x_2_3 = np.append(train_x ** 2, train_x ** 3, axis=1)
  # train_x_2_3_4 = np.append(train_x_2_3, train_x ** 4, axis=1)
  # train_x = np.append(train_x, train_x ** 2, axis=1)

  indices = np.random.permutation(train_x.shape[0])
  tra_id, val_id = indices[:5000], indices[5000:]
  tra_x, val_x = train_x[tra_id], train_x[val_id]
  tra_y, val_y = train_y[tra_id], train_y[val_id]

  return tra_x, tra_y, val_x, val_y


def cal_rmse(x, w, b, y):

  return np.sqrt(((y - b - (x*w).sum(axis=1).sum(axis=1)) ** 2).sum() / len(y))


def linear_regression(x, y, vx, vy):

  fNum = len(feat_list)
  lr = 0.01
  batch = 100
  print_every = 10
  iteration = 1000

  w = np.random.rand(fNum, x.shape[2]) / x.shape[0] / x.shape[1]
  b = 0

  b_lr = 0.0
  w_lr = np.zeros((fNum, x.shape[2]))

  b_history = np.array(b)
  w_history = np.array(w)

  print('Starting to train model')

  for i in range(iteration + 1):

    b_grad = 0.0
    w_grad = np.zeros((fNum, x.shape[2]))

    batchNum = 0

    for n in np.random.permutation(x.shape[0]):

      if batchNum < batch:

        batchNum += 1
        b_grad = b_grad - 2.0*(y[n] - b - (w*x[n]).sum())
        w_grad = w_grad - 2.0*(y[n] - b - (w*x[n]).sum())*x[n]

      else:

        b_lr += b_grad ** 2
        w_lr += w_grad ** 2
        b = b - lr/np.sqrt(b_lr / (i+1)) * b_grad
        w = w - lr/np.sqrt(w_lr / (i+1)) * w_grad
        b_grad = 0.0
        w_grad = np.zeros((fNum, x.shape[2]))
        batchNum = 0
        b_history = np.append(b_history, b)
        w_history = np.append(w_history, w)
        
    trmse = cal_rmse(x, w, b, y)
    vrmse = cal_rmse(vx, w, b, vy)

    if i % print_every == 0:

      print('Iteration: %d, TRMSE: %.4f, VRMSE: %.4f' % (i, trmse, vrmse))

  return w, b

def read_testing_data(name):

  print('Reading testing data')
  df = pd.read_csv(name, encoding='Big5', na_values='NR', header=None)
  
  # replace NR by 0
  df.fillna(value=0, inplace=True)

  data = {}
  
  # group data in date
  for fcol, ddf in df.groupby(0):

      idx = fcol.strip('id_')

      # drop id column in ddf
      ddf.drop(0, axis=1, inplace=True)

      # set parameter into row index
      ddf.set_index(1, drop=True, inplace=True)

      ddf.columns = [i for i in range(9)]

      if idx in data:
        data[idx] = pd.concat([data[idx], ddf], axis=1)
      else:
        data[idx] = ddf

  for key in data.keys():

    data[key] = data[key][data[key].columns.sort_values()]

  return data


def process_testing_data(data):

  test_x = []

  for i in range(len(data)):

    X = data[str(i)].values

    tmp_x = []

    for j in feat_list:

      tmp_x.append(X[j][0:9])

    test_x.append(tmp_x)

  test_x = np.array(test_x)

  # add sqaured data
  # train_x_2_3 = np.append(train_x ** 2, train_x ** 3, axis=1)
  # train_x_2_3_4 = np.append(train_x_2_3, train_x ** 4, axis=1)
  # train_x = np.append(train_x, train_x ** 2, axis=1)

  return test_x


def use_model(x, w, b):

  print('Using model to get prediction')
  y = (x*w).sum(axis=1).sum(axis=1) + b

  return y


def output_result(y, name):

  if not os.path.exists(os.path.dirname(name)):

    try:
      os.makedirs(os.path.dirname(name))

    except OSError as exc:
      if exc.errno != errno.EEXIST:
        raise

  file = open(name, 'w')
  file.write('id,value\n')

  for i in range(len(y)):

    file.write('id_%d,%d\n' % (i, y[i]))


def use_saved_model(x):

  w = np.load('./best_model/weight.npy')
  b = np.load('./best_model/bias.npy')
  y = use_model(x, w, b)
  output_result(y, output_name)


def main():

  if (len(sys.argv) == 5):

    test_data = read_testing_data(test_data_name)
    test_x = process_testing_data(test_data)
    use_saved_model(test_x)
    return

  train_data = read_training_data(train_data_name)
  train_x, train_y, val_x, val_y = process_training_data(train_data)
  weight, bias = linear_regression(train_x, train_y, val_x, val_y)

  print('Saving model')
  np.save('weight', weight)
  np.save('bias', bias)

  test_data = read_testing_data(test_data_name)
  test_x = process_testing_data(test_data)
  test_y = use_model(test_x, weight, bias)

  output_result(test_y, output_name)


if __name__ == '__main__':

  main()

