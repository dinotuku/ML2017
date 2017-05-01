#!/usr/bin/env python
# -- coding: utf-8 --
"""
Plot training information
"""

import os
import argparse
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from utils import loss

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def main():
    """ Main function """
    parser = argparse.ArgumentParser(prog='plot_model.py', description='Plot the model.')
    parser.add_argument('--model', type=str, default='simple',
                        choices=['simple', 'easy', 'strong', 'dnn', 'semi'], metavar='<model>')
    parser.add_argument('--epoch', type=int, default=100, metavar='<#epoch>')
    parser.add_argument('--dataGen', type=bool, default=False, metavar='<use_dataGen(bool)>')
    parser.add_argument('--idx', type=int, required=True, metavar='<suffix>')
    args = parser.parse_args()
    store_path = "{}_epoch{}{}_{}".format(args.model, args.epoch,
                                          ('_G' if args.dataGen else ''), args.idx)
    print("Loading model from {}".format(store_path))
    model_path = os.path.join(MODEL_DIR, store_path, 'final_model.hdf5')

    if (args.model == 'semi'):
        emotion_classifier = load_model(model_path, custom_objects={'loss': loss})
    else:
        emotion_classifier = load_model(model_path)

    emotion_classifier.summary()
    plot_model(emotion_classifier, to_file=os.path.join(MODEL_DIR, store_path, 'model.png'), show_shapes=True)

if __name__ == "__main__":

    main()
