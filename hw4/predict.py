# -*- coding: utf8 -*-
"""
Use linear SVR to predict intrinsic dimensions
"""

import sys
import numpy as np
from sklearn.svm import LinearSVR as SVR
from gen import get_eigenvalues

def main():
    """ Main function """
    # Train a linear SVR

    npzfile = np.load('./model/train_data.npz')
    x_data = npzfile['X']
    y_data = npzfile['y']

    svr = SVR(C=30)
    svr.fit(x_data, y_data)

    # Predict intrinsic dimensions
    test_data = np.load(sys.argv[1])
    test_x = []
    for i in range(200):
        data = test_data[str(i)]
        eigen_vals = get_eigenvalues(data)
        test_x.append(eigen_vals)

    test_x = np.array(test_x)
    pred_y = svr.predict(test_x)
    pred_y = np.rint(pred_y)
    np.place(pred_y, pred_y < 1, 1)

    with open(sys.argv[2], 'w') as f:
        print('SetId,LogDim', file=f)
        for idx, dim in enumerate(pred_y):
            print("{:d},{:f}".format(idx, np.log(dim)), file=f)

if __name__ == '__main__':

    main()
