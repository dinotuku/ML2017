# -*- coding: utf8 -*-
"""
Use linear SVR to predict intrinsic dimensions of hand rotation sequence dataset
"""

import os
import numpy as np
from PIL import Image
from sklearn.svm import LinearSVR as SVR
from hand_gen import get_eigenvalues

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    """ Main function """
    # Train a linear SVR

    npzfile = np.load('train_data_h.npz')
    x_data = npzfile['X']
    y_data = npzfile['y']

    svr = SVR(C=30)
    svr.fit(x_data, y_data)

    # Predict intrinsic dimensions
    test_data = []
    for num in range(481):
        image_path = os.path.join(BASE_DIR, 'data/handRotation',
                                  "hand.seq{:d}.png".format(num + 1))
        image = Image.open(image_path).resize((64, 64), resample=5)
        test_data.append(list(image.getdata()))

    test_data = np.array(test_data)
    test_x = []
    eigen_vals = get_eigenvalues(test_data)
    test_x.append(eigen_vals)

    test_x = np.array(test_x)
    pred_y = svr.predict(test_x)
    print(pred_y)
    pred_y = np.rint(pred_y)
    print(pred_y)
    np.place(pred_y, pred_y < 1, 1)

    with open('ans_h.csv', 'w') as f:
        print('SetId,LogDim', file=f)
        for idx, dim in enumerate(pred_y):
            print("{:d},{:f}".format(idx, np.log(dim)), file=f)

if __name__ == '__main__':

    main()
