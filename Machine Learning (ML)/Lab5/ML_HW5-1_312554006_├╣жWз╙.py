# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:26:22 2023

@author: max2306
"""

# the log likelihood function is referred to the following website:
#      https://stats.stackexchange.com/questions/280105/log-marginal-likelihood-for-gaussian-process
#      https://gaussianprocess.org/gpml/chapters/RW2.pdf


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys

def kernel(X_1, X_2, amplitude=1, rate=1, scale=1):
    # [X_n - X_m]^2
    dis = np.zeros((len(X_1), len(X_2)))
    for i in range(len(X_1)):
        for j in range(len(X_2)):
            dis[i][j] = np.power(X_1[i][0] - X_2[j][0], 2)
    return amplitude**2*(1+dis/(2*rate*(scale**2)))**-rate

# read the file


def read_file(file_name):

    X = []
    Y = []
    with open(file_name, 'r') as file:
        for line in file:
            x, y = map(float, line.split())
            X.append(x)
            Y.append(y)

    df = [[x, y] for x, y in zip(X, Y)]

    return df


def GPD(df, beta, amplitude, rate, scale):

    X = [x[0] for x in df]
    X = np.array(X).reshape(-1, 1)
    Y = [y[1] for y in df]
    Y = np.array(Y).reshape(-1, 1)
    X_new = np.linspace(-60, 60, 120).reshape(-1, 1)
    train_size = len(X)

    # follow the formula in the slide pg.50
    C = kernel(X, X) + 1/beta*np.identity(train_size)
    mean = kernel(X, X_new).T@np.linalg.inv(C)@Y
    var = kernel(X_new, X_new) + (1/beta)*np.identity(len(X_new)) - \
        kernel(X, X_new).T@np.linalg.inv(C)@kernel(X, X_new)

    var = np.diagonal(var).reshape(-1, 1)
    #plt.scatter(X, Y)
    plt.plot(X_new, mean, 'k-')
    plt.fill_between(np.sum(X_new, axis=1), np.sum(
        mean+1.96*np.sqrt(var), axis=1), np.sum(mean-1.96*np.sqrt(var), axis=1))
    plt.scatter(X, Y, c='r')
    plt.show()


def optimizer(params, df, beta):
    X = [x[0] for x in df]
    X = np.array(X).reshape(-1, 1)
    Y = [y[1] for y in df]
    Y = np.array(Y).reshape(-1, 1)
    train_size = len(X)
    C = kernel(X, X, params[0], params[1], params[2]) + \
        1/beta*np.identity(train_size)
    log_likelihood = (1/2)*Y.T@np.linalg.inv(C)@Y - (1/2)*np.sum(np.log(
        np.diagonal(np.linalg.cholesky(C)))) - (1/2)*train_size*np.log(2*np.pi)

    return -log_likelihood[0][0]


if __name__ == "__main__":

    df = read_file("./data./input.data.txt")

    # task 1 : gaussen process regression
    if sys.argv[1] == "1":
        beta = int(sys.argv[2])
        GPD(df, beta, int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))

    # task 2 : optimize
    else:        
        beta = int(sys.argv[2])
        param = minimize(optimizer, [int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])], args=(df, beta))
        amplitude = param.x[0]
        rate = param.x[1]
        scale = param.x[2]
        GPD(df, beta, amplitude, rate, scale)
        print(f"best amplitude {amplitude}")
        print(f"best rate {rate}")
        print(f"best scale {scale}")
