# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 20:49:27 2023

@author: max2306
"""

import numpy as np
# ref link: https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
# using An easy-to-program approximate approach


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
    x = np.expand_dims(x, axis=1)
    noise = univariate_generator(0, a)

    y = np.dot(w.T, x)[0][0] + noise
    return x[1][0], y


if __name__ == "__main__":

    # a. univariate gaussian data generator
    input_list = input().split()
    mean = float(input_list[0])
    var = float(input_list[1])
    ans = univariate_generator(mean, var)
    print(ans)

    # b. polynomial data generator
    n = int(input())
    a = float(input())
    w = np.zeros((n, 1))
    for i in range(n):
        w[i][0] = float(input())
    x, y = poly_generator(n, a, w)
    print(x, y)
