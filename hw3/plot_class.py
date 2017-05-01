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

np.set_printoptions(threshold=np.inf)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'image')
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)
FIL_DIR = os.path.join(IMG_DIR, 'vis_filter')
if not os.path.exists(FIL_DIR):
    os.makedirs(FIL_DIR)
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def normalize(x_inp):
    """ Utility function to normalize a tensor by its L2 norm """
    return x_inp / (K.sqrt(K.mean(K.square(x_inp))) + 1e-5)

def grad_ascent(num_step, input_image_data, iter_func):
    """ Perform gradient ascent """
    images = []
    losses = 0
    for _ in range(num_step):
        loss_value, grads_value = iter_func([input_image_data, 0])
        input_image_data += grads_value * 500 # step size
        losses += loss_value
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    images.append(input_image_data)
    images.append(losses)
    return images

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

    # layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    input_img = emotion_classifier.input

    name_ls = ['dense_3']
    # collect_layers = [layer_dict[name].output for name in name_ls]
    collect_layers = [emotion_classifier.output]
    num_step = 100

    for cnt, layer in enumerate(collect_layers):
        class_imgs = [[] for i in range(7)]
        for class_idx in range(7):
            input_img_data = np.random.random(size=(1, 48, 48, 1)) # random noise
            target = layer[:, class_idx]
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img, K.learning_phase()], [target, grads])
            class_imgs[class_idx] = grad_ascent(num_step, input_img_data, iterate)

        fig = plt.figure(figsize=(14, 4))
        pred_res = []
        for i in range(7):
            pred_res.append(emotion_classifier
                            .predict(class_imgs[i][0].reshape(1, 48, 48, 1)))
            print(pred_res[i])
            axis = fig.add_subplot(1, 7, i+1)
            axis.imshow(class_imgs[i][0].reshape(48, 48), cmap='gray')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel("Class {} {:.3f}%".format(np.argmax(np.array(pred_res), axis=2)[i][0],
                                                 max(pred_res[i][0])*100))
            plt.tight_layout()

        print(np.argmax(np.array(pred_res), axis=2))
        fig.suptitle("Filters of layer {} (# Ascent Epoch {} )".format(name_ls[cnt], num_step))
        img_path = os.path.join(FIL_DIR, '{}-{}'.format(store_path, name_ls[cnt]))
        if not os.path.isdir(img_path):
            os.mkdir(img_path)

        fig.savefig(os.path.join(img_path, 'e{}'.format(num_step)))

    K.clear_session()

if __name__ == "__main__":

    main()
