# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 23:53:38 2023

@author: max2306
"""
import numpy as np
import math
import matplotlib.pyplot as plt


def univariate_generator(mean, var):
    x = standard_generator()
    return mean + np.sqrt(var)*x


def standard_generator():
    total = 0
    for i in range(12):
        total += np.random.uniform(0, 1)
    return total - 6


def poly_generator(n, a, w):
    x = np.random.uniform(-1, 1)
    x = np.power(x, np.arange(n))
    noise = univariate_generator(0, a)

    y = np.dot(w.T, x)[0] + noise
    return x, y


def posterior(a, b, n):

    prior_mean = np.zeros((n, 1))
    prior_var = np.identity(n)*(1/b)

    epoch = 0
    data_x = []
    data_y = []
    mean_list = []
    var_list = []

    while(epoch < 1000):
        x, y = poly_generator(n, a, w)
        x = np.expand_dims(x, axis=0)
        data_x.append(x[0][1])
        data_y.append(y)
        print("Add data point (" + str(x[0][1]) + ", " + str(y) + "):")

        S = np.linalg.inv(prior_var)
        post_var = np.linalg.inv((1/a) * np.dot(x.T, x) + S)
        temp = ((1/a)*x.T*y+np.dot(S, prior_mean))
        post_mean = np.dot(post_var, temp)

        print("Posterior mean:")
        print(post_mean)
        print('\nPosterior variance:')
        print(post_var)

        pred_mean = np.dot(x, post_mean)
        pred_var = (a) + np.dot(np.dot(x, post_var), x.T)
        print("Predicitve distribution ~ N(" +
              str(pred_mean[0][0])+", "+str(pred_var[0][0])+")")

        prior_var = post_var
        prior_mean = post_mean
        if(epoch == 10 or epoch == 20):
            mean_list.append(prior_mean)
            var_list.append(prior_var)

        epoch += 1

    mean_list.append(prior_mean)
    var_list.append(prior_var)
    return mean_list, var_list, data_x, data_y


def plot(n, num_size, data_x, data_y, mean, var, a, title):
    x = np.linspace(-2, 2, 1000)
    pred_y = np.zeros((1000,))
    pred_var = np.zeros((1000,))
    for i in range(1000):
        x_i = np.power(x[i], np.arange(n))
        x_i = np.expand_dims(x_i, axis=0)
        pred_y[i] = np.dot(x_i, mean)
        pred_var[i] = (a) + np.dot(np.dot(x_i, var), x_i.T)[0][0]
    plt.plot(data_x[:num_size], data_y[:num_size], 'bo')
    plt.plot(x, pred_y, 'k-')
    plt.plot(x, pred_y+pred_var, 'r-')
    plt.plot(x, pred_y-pred_var, 'r-')
    plt.ylim(-20, 20)
    plt.title(title)
    plt.show()


if __name__ == "__main__":

    b = int(input())
    n = int(input())
    a = int(input())
    w = np.zeros((n, 1))
    for i in range(n):
        w[i][0] = int(input())

    # Online learning
    mean_list, var_list, data_x, data_y = posterior(a, b, n)

    # Ground truth
    plot(n, 0, data_x, data_y, w, np.zeros((n, n)), a, "Ground truth")
    # After 10 incomes
    plot(n, 10, data_x, data_y, mean_list[0],
         var_list[0], a, "After 10 incomes")
    # After 50 incomes
    plot(n, 50, data_x, data_y, mean_list[1],
         var_list[1], a, "After 50 incomes")
    # Predict result
    plot(n, 1000, data_x, data_y,
         mean_list[2], var_list[2], a, "Predict result")
