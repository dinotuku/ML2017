#!/usr/bin/env python
# -- coding: utf-8 --
"""
Plot the output of a filter when feeding in a image in validating set
"""

import os
import argparse
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np
from utils import read_dataset, loss

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'image')
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)
LAY_DIR = os.path.join(IMG_DIR, 'vis_layer')
if not os.path.exists(LAY_DIR):
    os.makedirs(LAY_DIR)
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def main():
    """ Main function """
    parser = argparse.ArgumentParser(prog='predict.py',
                                     description='ML-Assignment3 testing script.')
    parser.add_argument('--input', type=str, metavar='<input_file>', default='./data/test.csv')
    parser.add_argument('--model', type=str, default='easy',
                        choices=['easy', 'simple', 'strong', 'dnn', 'semi'], metavar='<model>')
    parser.add_argument('--epoch', type=int, default=100, metavar='<#epoch>')
    parser.add_argument('--dataGen', type=bool, default=False, metavar='<use_dataGen(bool)>')
    parser.add_argument('--batch', type=int, default=256, metavar='<batch_size>')
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

    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

    input_img = emotion_classifier.input
    name_ls = ['conv2d_2']
    collect_layers = [K.function([input_img, K.learning_phase()], [layer_dict[name].output])
                      for name in name_ls]

    private_pixels = read_dataset('./data/test.csv', False)
    private_pixels = [private_pixels[i].reshape((1, 48, 48, 1))
                      for i in range(len(private_pixels))]

    choose_id = 74
    photo = private_pixels[choose_id]
    for cnt, func in enumerate(collect_layers):
        img = func([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 12))
        nb_filter = img[0].shape[3]
        for i in range(nb_filter):
            axis = fig.add_subplot(nb_filter/16, 16, i+1)
            axis.imshow(img[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()

        fig.suptitle('Output of layer {} (Given image{})'.format(name_ls[cnt], choose_id))
        img_path = os.path.join(LAY_DIR, store_path)
        if not os.path.isdir(img_path):
            os.mkdir(img_path)

        fig.savefig(os.path.join(img_path, 'layer-{}'.format(name_ls[cnt])))

    K.clear_session()

if __name__ == '__main__':

    main()
