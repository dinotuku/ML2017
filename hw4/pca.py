"""
Eigenfaces with PCA
"""

import os
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'image', 'pca')
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

def q1_plot_ave(face):
    """ Plot the average face """
    fig = plt.figure()
    plt.title('Average face')
    plt.imshow(face.reshape(64, 64), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    fig.savefig(os.path.join(IMG_DIR, 'q1_ave'))
    fig.clear()


def q1_plot_eig(faces, evalues):
    """ Plot the first eigenfaces """
    fig = plt.figure(figsize=(9, 9))
    fig.suptitle('Top 9 eigenfaces', fontsize=16)
    for idx, face in enumerate(faces):
        axis = fig.add_subplot(3, 3, idx + 1)
        axis.imshow(face.reshape(64, 64), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(evalues[idx])
        plt.tight_layout()

    plt.subplots_adjust(top=0.95)
    fig.savefig(os.path.join(IMG_DIR, 'q1_eig'))
    fig.clear()

def q2_plot(faces, types):
    """ Plot the 100 original faces and the recovered faces """
    fig = plt.figure(figsize=(18, 18))
    fig.suptitle("{:s} faces".format(types), fontsize=20)
    for idx, face in enumerate(faces):
        axis = fig.add_subplot(10, 10, idx + 1)
        axis.imshow(face.reshape(64, 64), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

    plt.subplots_adjust(top=0.95)
    fig.savefig(os.path.join(IMG_DIR, 'q2_ori' if types == 'Original' else 'q2_rec'))
    fig.clear()

def main():
    """ Main function """
    parser = ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/faceExpressionDatabase',
                        help='Face expression images')
    args = parser.parse_args()
    # Q1
    print('Loading data')
    q1_faces = []
    for let in range(10):
        for num in range(10):
            image_path = os.path.join(BASE_DIR, args.data_path,
                                      "{:s}{:02d}.bmp".format(chr(let + 65), num))
            q1_faces.append(list(Image.open(image_path).getdata()))

    q1_faces = np.array(q1_faces)
    q1_faces_mean = q1_faces.mean(axis=0, keepdims=True)
    q1_faces_ctr = q1_faces - q1_faces_mean
    print('Perform SVD')
    _, eigenvalues, eigenfaces = np.linalg.svd(q1_faces_ctr)
    print(eigenfaces.shape)
    print('Plot faces of q1')
    q1_plot_ave(q1_faces_mean)
    q1_plot_eig(eigenfaces[:9], eigenvalues[:9])

    # Q2
    print('Project each face onto eigenfaces')
    q2_faces_prejected = np.matmul(q1_faces_ctr, eigenfaces[:5].transpose())
    print('Recover faces')
    q2_faces_recovered = q1_faces_mean + np.matmul(q2_faces_prejected, eigenfaces[:5])
    print('Plot faces of q2')
    q2_plot(q1_faces, 'Original')
    q2_plot(q2_faces_recovered, 'Recovered')

    # Q3
    print('Find the smallest number of eigenfaces')
    min_k = 0
    for k in range(eigenfaces.shape[0]):
        q3_faces_prejected = np.matmul(q1_faces_ctr, eigenfaces[:k + 1].transpose())
        q3_faces_recovered = q1_faces_mean + np.matmul(q3_faces_prejected, eigenfaces[: k + 1])
        rmse = np.sqrt(np.mean((q3_faces_recovered / 256 - q1_faces / 256) ** 2))
        if rmse < 0.01:
            min_k = k + 1
            break

    print("Smallest k = {:d}".format(min_k))
    print("RMSE = {:f}".format(rmse))

if __name__ == '__main__':

    main()
