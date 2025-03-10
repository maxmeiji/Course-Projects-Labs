# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:35:52 2023

@author: max2306
"""

import numpy as np


def read_file(tr_image, tr_label):

    # training images
    offset, tr_num, tr_row, tr_col = np.fromfile(
        file=tr_image, dtype=np.dtype('>i4'), count=4)
    tr_images = np.fromfile(file=tr_image, dtype=np.dtype('>B'))[16:]
    tr_pixels = tr_row * tr_col
    # 28*28*8
    tr_images = np.reshape(tr_images, (tr_num, tr_pixels))
    tr_images = np.asarray(tr_images >= 128, dtype='uint8')

    # traing label
    offset, tr_num = np.fromfile(file=tr_label, dtype=np.dtype('>i4'), count=2)
    tr_labels = np.fromfile(file=tr_label, dtype=np.dtype('>B'))[8:]

    return tr_images, tr_labels


def E_step(tr_images, prior, P):

    W = np.ones((60000, 10))
    for i in range(60000):
        for j in range(10):

            W[i][j] = np.prod(P[:, j]*tr_images[i, :] +
                              (1-P[:, j])*(1-tr_images[i, :]))

    W *= prior

    for i in range(60000):
        if np.sum(W[i][:] != 0):
            W[i][:] /= np.sum(W[i][:])

    return W


def M_step(tr_images, W):

    # update prior
    prior = np.sum(W, axis=0)
    prior /= 60000

    # update P
    norm = np.sum(W, axis=0)
    for i in range(10):
        if norm[i] != 0:
            W[:][i] /= norm[i]
    P = np.dot(tr_images.T, W)

    return prior, P

    return prior, P


if __name__ == " __init__":

    # input file
    tr_image = "train-images.txt"
    tr_label = "train-labels.txt"

    tr_images, tr_labels = read_file(tr_image, tr_label)

    # run E-M algorithms
    prior_init = np.random.rand(10)
    prior_init /= np.sum(prior_init)
    P_init = np.random.rand(784, 10)
    epoch = 1
    while epoch < 100:
        # E
        W = E_step(tr_images, prior_init, P_init)
        # M
        prior, P = M_step(tr_images, W)

        if (np.sum(abs(prior - prior_init)) + np.sum(abs(P-P_init)) < 784):
            break
        else:
            prior_init = prior
            P_init = P
            epoch += 1
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:35:52 2023

@author: max2306
"""


def read_file(tr_image, tr_label):

    # training images
    offset, tr_num, tr_row, tr_col = np.fromfile(
        file=tr_image, dtype=np.dtype('>i4'), count=4)
    tr_images = np.fromfile(file=tr_image, dtype=np.dtype('>B'))[16:]
    tr_pixels = tr_row * tr_col
    # 28*28*8
    tr_images = np.reshape(tr_images, (tr_num, tr_pixels))
    tr_images = np.asarray(tr_images >= 128, dtype='uint8')

    # traing label
    offset, tr_num = np.fromfile(file=tr_label, dtype=np.dtype('>i4'), count=2)
    tr_labels = np.fromfile(file=tr_label, dtype=np.dtype('>B'))[8:]

    return tr_images, tr_labels


def E_step(tr_images, prior, P):

    W = np.ones((60000, 10))
    for i in range(60000):
        for j in range(10):

            W[i][j] = np.prod(P[:, j]*tr_images[i, :] +
                              (1-P[:, j])*(1-tr_images[i, :]))

    W = W*prior.reshape(1, -1)

    for i in range(60000):
        if np.sum(W[i][:]) != 0:

            W[i][:] /= np.sum(W[i][:])

    return W


def M_step(tr_images, W):

    # update prior
    prior = np.sum(W, axis=0)
    prior /= np.sum(prior)

    # update P

    P = np.dot(tr_images.T, W)
    sums = np.sum(P, axis=1).reshape(-1, 1)
    sums[sums == 0] = 1
    P = P / sums

    return prior, P


def plot(tr_images, tr_labels, P):
    for i in range(10):
        print("class "+str(i) + ": ")
        print(tr_images[i])

        print("label "+str(i) + ": ")
        print(tr_labels[i])


if __name__ == "__main__":

    # input file
    tr_image = "train-images.txt"
    tr_label = "train-labels.txt"

    tr_images, tr_labels = read_file(tr_image, tr_label)

    # run E-M algorithms
    prior_init = np.random.rand(10)
    prior_init /= np.sum(prior_init)
    P_init = np.random.rand(784, 10)
    epoch = 1
    while epoch < 10:
        # E
        W = E_step(tr_images, prior_init, P_init)
        # M
        prior, P = M_step(tr_images, W)

        if (np.sum(abs(prior - prior_init)) + np.sum(abs(P-P_init)) < 784):
            break
        else:
            prior_init = prior
            P_init = P
            epoch += 1

        print(epoch)
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:35:52 2023

@author: max2306
"""


def read_file(tr_image, tr_label):

    # training images
    offset, tr_num, tr_row, tr_col = np.fromfile(
        file=tr_image, dtype=np.dtype('>i4'), count=4)
    tr_images = np.fromfile(file=tr_image, dtype=np.dtype('>B'))[16:]
    tr_pixels = tr_row * tr_col
    # 28*28*8
    tr_images = np.reshape(tr_images, (tr_num, tr_pixels))
    tr_images = np.asarray(tr_images >= 128, dtype='uint8')

    # traing label
    offset, tr_num = np.fromfile(file=tr_label, dtype=np.dtype('>i4'), count=2)
    tr_labels = np.fromfile(file=tr_label, dtype=np.dtype('>B'))[8:]

    return tr_images, tr_labels


def E_step(tr_images, prior, P):

    W = np.ones((60000, 10))
    for i in range(60000):
        for j in range(10):

            W[i][j] = np.prod(P[:, j]*tr_images[i, :] +
                              (1-P[:, j])*(1-tr_images[i, :]))

    W *= prior

    for i in range(60000):
        if np.sum(W[i][:] != 0):
            W[i][:] /= np.sum(W[i][:])

    return W


def M_step(tr_images, W):

    # update prior
    prior = np.sum(W, axis=0)
    prior /= 60000

    # update P
    norm = np.sum(W, axis=0)
    for i in range(10):
        if norm[i] != 0:
            W[:][i] /= norm[i]
    P = np.dot(tr_images.T, W)

    return prior, P

    return prior, P


if __name__ == " __init__":

    # input file
    tr_image = "train-images.txt"
    tr_label = "train-labels.txt"

    tr_images, tr_labels = read_file(tr_image, tr_label)

    # run E-M algorithms
    prior_init = np.random.rand(10)
    prior_init /= np.sum(prior_init)
    P_init = np.random.rand(784, 10)
    epoch = 1
    while epoch < 100:
        # E
        W = E_step(tr_images, prior_init, P_init)
        # M
        prior, P = M_step(tr_images, W)

        if (np.sum(abs(prior - prior_init)) + np.sum(abs(P-P_init)) < 784):
            break
        else:
            prior_init = prior
            P_init = P
            epoch += 1
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:35:52 2023

@author: max2306
"""


def read_file(tr_image, tr_label):

    # training images
    offset, tr_num, tr_row, tr_col = np.fromfile(
        file=tr_image, dtype=np.dtype('>i4'), count=4)
    tr_images = np.fromfile(file=tr_image, dtype=np.dtype('>B'))[16:]
    tr_pixels = tr_row * tr_col
    # 28*28*8
    tr_images = np.reshape(tr_images, (tr_num, tr_pixels))
    tr_images = np.asarray(tr_images >= 128, dtype='uint8')

    # traing label
    offset, tr_num = np.fromfile(file=tr_label, dtype=np.dtype('>i4'), count=2)
    tr_labels = np.fromfile(file=tr_label, dtype=np.dtype('>B'))[8:]

    return tr_images, tr_labels


def E_step(tr_images, prior, P):

    W = np.ones((60000, 10))
    for i in range(60000):
        for j in range(10):

            W[i][j] = np.prod(P[:, j]*tr_images[i, :] +
                              (1-P[:, j])*(1-tr_images[i, :]))

    W = W*prior.reshape(1, -1)

    for i in range(60000):
        if np.sum(W[i][:]) != 0:

            W[i][:] /= np.sum(W[i][:])

    return W


def M_step(tr_images, W):

    # update prior
    prior = np.sum(W, axis=0)
    prior /= np.sum(prior)

    # update P

    P = np.dot(tr_images.T, W)
    sums = np.sum(P, axis=1).reshape(-1, 1)
    sums[sums == 0] = 1
    P = P / sums

    return prior, P


def plot(tr_images, tr_labels, P):
    for i in range(10):
        print("class "+str(i) + ": ")
        print(tr_images[i])

        print("label "+str(i) + ": ")
        print(tr_labels[i])


if __name__ == "__main__":

    # input file
    tr_image = "train-images.txt"
    tr_label = "train-labels.txt"

    tr_images, tr_labels = read_file(tr_image, tr_label)

    # run E-M algorithms
    prior_init = np.random.rand(10)
    prior_init /= np.sum(prior_init)
    P_init = np.random.rand(784, 10)
    epoch = 1
    while epoch < 10:
        # E
        W = E_step(tr_images, prior_init, P_init)
        # M
        prior, P = M_step(tr_images, W)

        if (np.sum(abs(prior - prior_init)) + np.sum(abs(P-P_init)) < 784):
            break
        else:
            prior_init = prior
            P_init = P
            epoch += 1

        print(epoch)
    plot(tr_images, tr_labels, P)
