#!/usr/bin/env python
# -- coding: utf-8 --
"""
Plot saliency figure
"""

import os
import copy
import argparse
import keras.backend as K
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from utils import read_dataset, loss

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'image')
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)
CMAP_DIR = os.path.join(IMG_DIR, 'cmap')
if not os.path.exists(CMAP_DIR):
    os.makedirs(CMAP_DIR)
PARTIAL_SEE_DIR = os.path.join(IMG_DIR, 'partial_see')
if not os.path.exists(PARTIAL_SEE_DIR):
    os.makedirs(PARTIAL_SEE_DIR)
SEE_DIR = os.path.join(IMG_DIR, 'see')
if not os.path.exists(SEE_DIR):
    os.makedirs(SEE_DIR)
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def main():
    """ Main function """
    parser = argparse.ArgumentParser(prog='plot_sali.py',
                                     description='Visualize attention heat map.')
    parser.add_argument('--model', type=str, default='simple',
                        choices=['simple', 'easy', 'strong', 'dnn', 'semi'], metavar='<model>')
    parser.add_argument('--epoch', type=int, default=100, metavar='<#epoch>')
    parser.add_argument('--dataGen', type=bool, default=False, metavar='<use_dataGen(bool)>')
    parser.add_argument('--idx', type=int, required=True, metavar='<suffix>')
    parser.add_argument('--choice', type=str, required=True, metavar='<model_choice>')
    args = parser.parse_args()
    store_path = "{}_epoch{}{}_{}".format(args.model, args.epoch,
                                          ('_G' if args.dataGen else ''), args.idx)
    print("Loading model from {}".format(store_path))
    model_path = os.path.join(MODEL_DIR, store_path, '{}_model.hdf5'.format(args.choice))
    if (args.model == 'semi'):
        emotion_classifier = load_model(model_path, custom_objects={'loss': loss})
    else:
        emotion_classifier = load_model(model_path)


    private_pixels = read_dataset('./data/test.csv', False)
    private_pixels = [private_pixels[i].reshape((1, 48, 48, 1))
                      for i in range(len(private_pixels))]
    input_img = emotion_classifier.input
    # img_ids = [202, 199, 74, 642, 1327, 71, 903]
    img_ids = [0]

    for idx in img_ids:
        val_proba = emotion_classifier.predict(private_pixels[idx])
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred[0]])
        grads = K.gradients(target, input_img)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        func = K.function([input_img, K.learning_phase()], [grads])

        heatmap = func([private_pixels[idx], 0])[0].reshape(2304)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        thres = 0.5
        par_see = private_pixels[idx].reshape(2304)
        see = copy.deepcopy(par_see)

        par_see[np.where(heatmap <= thres)] = np.mean(par_see)

        heatmap = heatmap.reshape(48, 48)
        par_see = par_see.reshape(48, 48)
        see = see.reshape(48, 48)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(CMAP_DIR, '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(par_see, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(PARTIAL_SEE_DIR, '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(SEE_DIR, '{}.png'.format(idx)), dpi=100)

    K.clear_session()

if __name__ == "__main__":
    main()
