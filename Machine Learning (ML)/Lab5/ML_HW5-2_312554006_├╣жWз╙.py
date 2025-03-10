# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:45:08 2023

@author: max2306
"""

import numpy as np
from libsvm.svmutil import *
import sys


def read_file(file_name):
    return np.genfromtxt(file_name, delimiter=",")

# the way to calculate distance is referenced by https://medium.com/swlh/euclidean-distance-matrix-4c3e1378d87f
def kernel(X_1, X_2, alpha, beta, gamma):
    X_1 = np.array(X_1)
    X_2 = np.array(X_2)
    linear = X_1@X_2.T
    dis = np.sum(X_1 ** 2, axis=1).reshape(-1, 1) + \
        np.sum(X_2 ** 2, axis=1) - 2 * X_1 @ X_2.T
    rbf = np.exp(-gamma*(dis))

    return np.multiply(linear, alpha) + np.multiply(rbf, beta)


if __name__ == "__main__":

    # readthe file
    X_train = read_file("./data/X_train.csv").tolist()
    X_test = read_file("./data/X_test.csv").tolist()
    Y_train = read_file("./data/Y_train.csv").tolist()
    Y_test = read_file("./data/Y_test.csv").tolist()

    if sys.argv[1] == "1":
        # task 1

        # svm model
        acc_list = []
        kernel_dict = {
            0: "linear",
            1: "polynomial",
            2: "RBF"
        }
        for i in range(3):
            model = svm_train(Y_train, X_train, '-s 0 -c 1 -t {}'.format(i))
            __, acc, __ = svm_predict(Y_test, X_test, model)
            acc_list.append(acc[0])
            print(f"accuracy of {kernel_dict[i]} kernel: {acc_list[i]}")

        # easier to demo the result
        print("summary: ")
        for i in range(3):
            print(f"accuracy of {kernel_dict[i]} kernel: {acc_list[i]}")

    elif sys.argv[1] == "2":
        # task 2
        # since the result in pamrt 1 shows the RBF is the best svm model,
        # use it to demonstrate grid search

        param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'gamma' : [0.01, 0.1, 1, 10, 100]}
        best_acc = 0
        # perform grid search and record the best accuracy combination
        for c in param_grid['C']:
            for g in param_grid['gamma']:
            # five fold cross-entropy
                params = '-s {} -c {} -t {} -g {} -v {}'.format(0, c, 2, g, 5)
                acc = svm_train(Y_train, X_train, params)
                if acc > best_acc:
                    best_acc = acc
                    best_param = [c, g]

        print(
            f"the best parameter c(cost) of RBF kernel is: {best_param[0]}")
        print(
            f"the best parameter g(gamma) of RBF kernel is: {best_param[1]}")
        print(f"the best accuracy of linear kernel is: {best_acc}")

    else:
        # task 3 - hybrid kernel
        # use t 4 to specify user-defined kernel
        kernel_matrix = kernel(X_train, X_train, 0.5, 0.5, 0.01)
        model = svm_train(Y_train, kernel_matrix, '-c 100 -t 4')
        __, p_acc, __ = svm_predict(Y_test, X_test, model)
        print(f"the accuracy of hybrid kernel is {p_acc[0]}")
