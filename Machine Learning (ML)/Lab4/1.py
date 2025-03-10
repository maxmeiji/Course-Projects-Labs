# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:42:35 2023

@author: max2306
"""

import numpy as np
import matplotlib.pyplot as plt


def univariate_generator(mean, var):
    x = standard_generator()
    return mean + np.sqrt(var)*x


def standard_generator():
    total = 0
    for i in range(12):
        total += np.random.uniform(0, 1)
    return total - 6


def input_func():
    # input
    N = int(input())
    mean_var = input()
    mean_var = str.split(mean_var, ",")
    mean_1 = []
    var_1 = []
    mean_2 = []
    var_2 = []

    for i in range(len(mean_var)//2):
        if i < 2:
            mean_1.append(int(mean_var[i*2]))
            var_1.append(int(mean_var[i*2+1]))
        else:
            mean_2.append(int(mean_var[i*2]))
            var_2.append(int(mean_var[i*2+1]))

    return N, mean_1, var_1, mean_2, var_2


def generating(N, mean_1, var_1, mean_2, var_2):
    Dx1 = []
    Dy1 = []
    Dx2 = []
    Dy2 = []

    for i in range(int(N)):
        Dx1.append(univariate_generator(mean_1[0], var_1[0]))
        Dy1.append(univariate_generator(mean_1[1], var_1[1]))
        Dx2.append(univariate_generator(mean_2[0], var_2[0]))
        Dy2.append(univariate_generator(mean_2[1], var_2[1]))

    return Dx1, Dy1, Dx2, Dy2


def matrix(N, Dx1, Dy1, Dx2, Dy2):

    y = np.zeros((1, 2*N))
    y[:, N:] = np.ones((1, N))

    w = np.ones((3, 1))

    x = np.ones((3, 2*N))
    x[1, :N] = Dx1
    x[1, N:] = Dx2
    x[2, :N] = Dy1
    x[2, N:] = Dy2

    return y, w, x


def steepest(w, y, x):
    print("steepest gradient descent: ")
    lr = 0.001
    while 1 == 1:
        d = np.dot((y-1/(1+np.exp(np.dot(-w.T, x)))), -x.T)
        w_new = w - lr*d.T
        if(np.sum(abs(w_new - w)) < 0.001):
            break
        else:
            w = w_new
    return w


def Newton(N, w, y, x):
    D = np.zeros((2*N, 2*N))
    for i in range(2*N):
        D[i][i] = np.exp(np.dot(-w.T, x[:, i])) / \
            np.power((1+np.exp(np.dot(-w.T, x[:, i]))), 2)
    H = np.dot(x, np.dot(D, x.T))
    try:
        H_inv = np.linalg.inv(H)
    except:
        print("H is not invertible, using steepest gradient method")
        return steepest(w, y, x)

    print("Newton's method: ")
    lr = 0.001
    while 1 == 1:
        d = np.dot((y-1/(1+np.exp(np.dot(-w.T, x)))), -x.T)
        w_new = w - lr*np.dot(H_inv, d.T)
        if(np.sum(abs(w_new - w)) < 0.001):
            break
        else:
            w = w_new
    return w


def output(N, w, y, x):

    # weight
    print("w: ")
    print(w[:, 0])

    # confusion matrix
    predict = 1/(1+np.exp(np.dot(-w.T, x)))
    for i in range(len(predict[0])):
        predict[0][i] = 1 if predict[0][i] > 0.5 else 0

    TP = np.count_nonzero(predict[0][:N] == 0)
    TN = np.count_nonzero(predict[0][N:] == 1)
    FN = N - TP
    FP = N - TN

    print("Confusion matrix : ")
    print('               Predict cluster 1  Predict cluster 2')
    print(
        'Is cluster 1        {:.0f}               {:.0f}       '.format(TP, FN))
    print(
        'Is cluster 2        {:.0f}               {:.0f}       '.format(FP, TN))

    print("\nSensitivity (Successfully predict cluster 1): " + str(TP/(TP+FN)))
    print("Specificity (Successfully predict cluster 2): " + str(format(TN/(TN+FP))))
    print("--------------------------------------------------------------------")
    return predict


def plot(predict_g, predict_N, Dx1, Dy1, Dx2, Dy2):
    # plot
    # Truth
    plt.scatter(Dx1, Dy1, color='red')
    plt.scatter(Dx2, Dy2, color='blue')
    plt.title("Ground Truth")
    plt.show()

    # Gradient descent
    ones = np.where(predict_g == 1)[1]
    zeros = np.where(predict_g == 0)[1]
    x_list = np.array(Dx1 + Dx2)
    y_list = np.array(Dy1 + Dy2)
    plt.scatter(x_list[zeros], y_list[zeros], color='red')
    plt.scatter(x_list[ones], y_list[ones], color='blue')
    plt.title("Gradient Dscent ")
    plt.show()

    # Newton's method
    ones = np.where(predict_N == 1)[1]
    zeros = np.where(predict_N == 0)[1]
    x_list = np.array(Dx1 + Dx2)
    y_list = np.array(Dy1 + Dy2)
    plt.scatter(x_list[zeros], y_list[zeros], color='red')
    plt.scatter(x_list[ones], y_list[ones], color='blue')
    plt.title("Gradient Dscent ")
    plt.show()


if __name__ == "__main__":

    # input
    N, mean_1, var_1, mean_2, var_2 = input_func()

    # generating data points
    Dx1, Dy1, Dx2, Dy2 = generating(N, mean_1, var_1, mean_2, var_2)

    # get matrix form
    y, w, x = matrix(N, Dx1, Dy1, Dx2, Dy2)

    # steepest gradient decent
    w_steep = steepest(w, y, x)
    predict_g = output(N, w_steep, y, x)

    # Newton's method
    w_Newton = Newton(N, w, y, x)
    predict_N = output(N, w_Newton, y, x)

    # plot
    plot(predict_g, predict_N, Dx1, Dy1, Dx2, Dy2)
