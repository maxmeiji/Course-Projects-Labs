# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 23:14:12 2023

@author: Max Lo

school id : 312554006

department : Computer Science

Institute : Data Science and Engineering
"""
import numpy as np


def discrete(tr_images, tr_labels, te_images, te_labels):

    # P = P(position, gray level | label) * P(label) / marginal

    # prior, P(label)
    prior = prioritize(tr_labels)

    # likelihood P(position, gray level | label)
    likelihood = likelize(tr_images, tr_labels)

    # log posterior
    num_test = len(te_images)
    pixels = te_images.shape[1]
    post_list = []
    pred_list = []
    err = 0

    for i in range(num_test):
        posterior = np.zeros((10,))
        for j in range(10):
            for k in range(pixels):
                posterior[j] += np.log(likelihood[j, k, te_images[i][k]//8])
            posterior[j] += np.log(prior[j])
        # normalize
        posterior /= sum(posterior)

        # prediction
        predict = np.argmin(posterior)
        if(predict != te_labels[i]):
            err += 1

        # store results
        pred_list.append(predict)
        post_list.append(posterior)

    result(post_list, pred_list, te_images,
           te_labels, err/(num_test), likelihood, 0)


def continous(tr_images, tr_labels, te_images, te_labels):

    # P = P(position, gray level | label) * P(label) / marginal

    # prior
    prior = prioritize(tr_labels)

    # likelihood
    mean_mle, var_mle = likelize_cont(tr_images, tr_labels)

    # posterior
    num_test = len(te_images)
    pixels = te_images.shape[1]
    post_list = []
    pred_list = []
    err = 0

    for i in range(num_test):
        posterior = np.zeros((10,))
        for j in range(10):
            for k in range(pixels):
                var = var_mle[j][k]
                mean = mean_mle[j][k]
                if var == 0:
                    continue
                posterior[j] -= np.log(np.sqrt(2*np.pi*var))
                posterior[j] -= np.power((te_images[i][k]-mean), 2)/(2*var)
            posterior[j] += np.log(prior[j])
        # normalize
        posterior /= sum(posterior)

        # prediction
        predict = np.argmin(posterior)
        if(predict != te_labels[i]):
            err += 1

        # store results
        pred_list.append(predict)
        post_list.append(posterior)

    result(post_list, pred_list, te_images,
           te_labels, err/(num_test), mean_mle, 1)


def prioritize(tr_labels):

    prior = np.zeros((10,))
    for i in range(len(tr_labels)):
        prior[tr_labels[i]] += 1
    prior /= len(tr_labels)

    return prior


def likelize(tr_images, tr_labels):
    # fit gray levels into 32 categories
    pixels = tr_images.shape[1]
    likelihood = np.zeros((10, pixels, 32))
    for i in range(len(tr_images)):
        for j in range(pixels):
            likelihood[tr_labels[i], j, tr_images[i][j]//8] += 1

    for i in range(10):
        normal = np.sum(likelihood[i])
        likelihood[i, :, :] /= normal

    likelihood[likelihood == 0] = 0.001

    return likelihood


def likelize_cont(tr_images, tr_labels):

    pixels = tr_images.shape[1]
    label_num = np.zeros((10, ))
    total = np.zeros((10, pixels))
    total_square = np.zeros((10, pixels))
    mean_mle = np.zeros((10, pixels))
    var_mle = np.zeros((10, pixels))

    for i in range(len(tr_images)):
        target = tr_labels[i]
        label_num[target] += 1
        for j in range(pixels):
            total[target][j] += tr_images[i][j]
            total_square[target][j] += np.power(tr_images[i][j], 2)
    for i in range(10):
        for j in range(pixels):
            mean_mle[i][j] = total[i][j] / label_num[i]
            var_mle[i][j] = total_square[i][j] / \
                label_num[i] - np.power(mean_mle[i][j], 2)

    return mean_mle, var_mle


def result(post_list, pred_list, te_images, te_labels, err_rate, likelihood, mode):

    # print out the posterior
    for i in range(len(post_list)):
        print("Posterior (in log scale): ")
        for j in range(10):
            print(str(j) + ": " + str(post_list[i][j]))
        print("Prediction: " +
              str(pred_list[i]) + ", Ans: " + str(te_labels[i]))
        print("\n")
        if(i > 3):
            break

    # imagination
    if mode == 0:
        pixels = te_images.shape[1]
        row = int(np.sqrt(pixels))
        for i in range(10):
            print(str(i) + ": ")
            for j in range(pixels):
                if(j % (row) == 0 and j != 0):
                    print('\n')
                pred_level = np.argmax(likelihood[i, j, :])
                if pred_level <= 15:
                    print("0", end=' ')
                else:
                    print("1", end=' ')

            print("\n")

    else:
        pixels = te_images.shape[1]
        row = int(np.sqrt(pixels))
        for i in range(10):
            print(str(i) + ": ")
            for j in range(pixels):
                if(j % (row) == 0 and j != 0):
                    print('\n')
                pred_level = likelihood[i][j]
                if pred_level <= 128:
                    print("0", end=' ')
                else:
                    print("1", end=' ')
            print("\n")

    # error rate
    print("Error rate: " + str(err_rate))


def read_file(tr_image, tr_label, te_image, te_label):

    # training images
    offset, tr_num, tr_row, tr_col = np.fromfile(
        file=tr_image, dtype=np.dtype('>i4'), count=4)
    tr_images = np.fromfile(file=tr_image, dtype=np.dtype('>B'))[16:]
    tr_pixels = tr_row * tr_col
    # 28*28*8
    tr_images = np.reshape(tr_images, (tr_num, tr_pixels))

    # testing images
    offset, te_num, te_row, te_col = np.fromfile(
        file=te_image, dtype=np.dtype('>i4'), count=4)
    te_images = np.fromfile(file=te_image, dtype=np.dtype('>B'))[16:]
    te_pixels = te_row * te_col
    # 28*28*8
    te_images = np.reshape(te_images, (te_num, te_pixels))

    # traing label
    offset, tr_num = np.fromfile(file=tr_label, dtype=np.dtype('>i4'), count=2)
    tr_labels = np.fromfile(file=tr_label, dtype=np.dtype('>B'))[8:]

    # testing label
    offset, te_num = np.fromfile(file=te_label, dtype=np.dtype('>i4'), count=2)
    te_labels = np.fromfile(file=te_label, dtype=np.dtype('>B'))[8:]

    return tr_images, tr_labels, te_images, te_labels


if __name__ == "__main__":

    # define file name and read file
    tr_image = "train-images.txt"
    tr_label = "train-labels.txt"
    te_image = "test-images.txt"
    te_label = "test-labels.txt"
    tr_images, tr_labels, te_images, te_labels = read_file(
        tr_image, tr_label, te_image, te_label)

    # discrete and continous
    mode = 1
    if mode == 0:
        discrete(tr_images, tr_labels, te_images, te_labels)
    else:
        continous(tr_images, tr_labels, te_images, te_labels)
