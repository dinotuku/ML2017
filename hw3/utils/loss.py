#!/usr/bin/env python
# -- coding: utf-8 --
"""
Functions for Loss
"""

import keras.losses

def loss(y_true, y_pred):
    """ Keras losses function """
    cross = keras.losses.categorical_crossentropy(y_true, y_pred)
    return cross
