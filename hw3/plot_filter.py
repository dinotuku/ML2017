#!/usr/bin/env python
# -- coding: utf-8 --
"""
Plot images that activate the selected filters the most
"""

import os
import argparse
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np
from utils import loss

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'image')
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)
FIL_DIR = os.path.join(IMG_DIR, 'vis_filter')
if not os.path.exists(FIL_DIR):
    os.makedirs(FIL_DIR)
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def normalize(x_inp):
    """ utility function to normalize a tensor by its L2 norm """
    return x_inp / (K.sqrt(K.mean(K.square(x_inp))) + 1e-7)

def grad_ascent(num_step, input_image_data, iter_func):
    """
    Implement this function!
    """
    filter_images = []
    losses = 0
    for _ in range(num_step):
        loss_value, grads_value = iter_func([input_image_data, 1])
        input_image_data += grads_value * 1 # step size
        losses += loss_value
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    filter_images.append(input_image_data)
    filter_images.append(losses)
    return filter_images

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
    collect_layers = [layer_dict[name].output for name in name_ls]
    num_step = 100

    for cnt, layer in enumerate(collect_layers):
        filter_imgs = [[] for i in range(64)]
        for filter_idx in range(64):
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            target = K.mean(layer[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img, K.learning_phase()], [target, grads])

            filter_imgs[filter_idx] = grad_ascent(num_step, input_img_data, iterate)

        fig = plt.figure(figsize=(14, 8))
        for i in range(64):
            axis = fig.add_subplot(64/16, 16, i+1)
            axis.imshow(filter_imgs[i][0].reshape(48, 48), cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            # plt.xlabel("{:.3f}".format(filter_imgs[i][1] / num_step))
            plt.tight_layout()

        fig.suptitle("Filters of layer {} (# Ascent Epoch {} )".format(name_ls[cnt], num_step))
        img_path = os.path.join(FIL_DIR, '{}-{}'.format(store_path, name_ls[cnt]))
        if not os.path.isdir(img_path):
            os.mkdir(img_path)

        fig.savefig(os.path.join(img_path, 'e{}'.format(num_step)))

    K.clear_session()

if __name__ == "__main__":

    main()
