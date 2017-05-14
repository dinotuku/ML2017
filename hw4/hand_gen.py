# -*- coding: utf8 -*-
"""
Generate training data for linear SVR of hand rotation sequence dataset
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def elu(arr):
    """ ELU activation function """
    return np.where(arr > 0, arr, np.exp(arr) - 1)

def make_layer(in_size, out_size):
    """ Create random weights and biases """
    weight = np.random.normal(scale=0.5, size=(in_size, out_size))
    bias = np.random.normal(scale=0.5, size=out_size)
    return (weight, bias)

def forward(inpd, layers):
    """ Calculate layer output """
    out = inpd
    for layer in layers:
        weight, bias = layer
        out = elu(np.matmul(out, weight) + bias)

    return out

def generate_data(dim, layer_dims, num):
    """ Generate data of different dimensions """
    layers = []
    data = np.random.normal(size=(num, dim))

    ndim = dim
    for dim in layer_dims:
        layers.append(make_layer(ndim, dim))
        ndim = dim

    weight, bias = make_layer(ndim, ndim)
    gen_data = forward(data, layers)
    gen_data = np.matmul(gen_data, weight) + bias
    return gen_data

def get_eigenvalues(data):
    """ Generate average eigenvalues """
    sample = 20
    neighbor = 20
    randidx = np.random.permutation(data.shape[0])[:sample]
    knbrs = NearestNeighbors(n_neighbors=neighbor,
                             algorithm='ball_tree').fit(data)

    sing_vals = []
    for idx in randidx:
        _, ind = knbrs.kneighbors(data[idx:idx+1])
        nbrs = data[ind[0, 1:]]
        _, eigen_vals, _ = np.linalg.svd(nbrs - nbrs.mean(axis=0))
        eigen_vals /= eigen_vals.max()
        sing_vals.append(eigen_vals)
    sing_vals = np.array(sing_vals).mean(axis=0)
    return sing_vals

def main():
    """ Main function """
    x_data = []
    y_data = []
    for i in range(100):
        dim = i + 1
        layer_dims = [np.random.randint(100, 3000), 4096]
        data = generate_data(dim, layer_dims, 481).astype('float32')
        eigenvalues = get_eigenvalues(data)
        x_data.append(eigenvalues)
        y_data.append(dim)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    np.savez('train_data_h.npz', X=x_data, y=y_data)

if __name__ == '__main__':

    main()
