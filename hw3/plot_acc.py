#!/usr/bin/env python
# -- coding: utf-8 --
"""
Plot accuracy and loss figure
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def plot_accuracy_figure(train, valid, mode, epoch):
    """
    This function prints and plots accuracy and loss figure
    """
    epoch = np.arange(epoch)
    plt.plot(epoch, train, label='train', color='#428bca')
    plt.plot(epoch, valid, label='valid', color='#d9534f')
    plt.legend()
    plt.grid()
    plt.title('Accuracy plot of CNN model' if mode == 'accuracy' else 'Loss plot of CNN model')
    plt.ylabel('Accuracy' if mode == 'accuracy' else 'Loss')
    plt.xlabel('# of epoch')

def main():
    """ Main function """
    parser = argparse.ArgumentParser(prog='plot_acc.py',
                                     description='Plot accuracy and loss figure.')
    parser.add_argument('--model', type=str, default='simple',
                        choices=['simple', 'easy', 'strong', 'dnn', 'semi'], metavar='<model>')
    parser.add_argument('--epoch', type=int, default=100, metavar='<#epoch>')
    parser.add_argument('--dataGen', type=bool, default=False, metavar='<use_dataGen(bool)>')
    parser.add_argument('--mode', type=str, default='accuracy',
                        choices=['accuracy', 'loss'], metavar='<mode>')
    parser.add_argument('--idx', type=int, required=True, metavar='<suffix>')
    args = parser.parse_args()
    store_path = "{}_epoch{}{}_{}".format(args.model, args.epoch,
                                          ('_G' if args.dataGen else ''), args.idx)
    print("Loading data from {}".format(store_path))
    train_path = os.path.join(MODEL_DIR, store_path, "train_{}".format(args.mode))
    valid_path = os.path.join(MODEL_DIR, store_path, "valid_{}".format(args.mode))

    train = np.loadtxt(train_path)
    valid = np.loadtxt(valid_path)

    plt.figure()
    plot_accuracy_figure(train, valid, args.mode, 40)
    plt.show()

if __name__ == "__main__":

    main()
