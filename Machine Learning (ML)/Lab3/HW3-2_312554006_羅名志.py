# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 22:11:41 2023

@author: max2306
"""

import numpy as np


def univariate_generator(mean, var):
    x = standard_generator()
    return mean + np.sqrt(var)*x


def standard_generator():
    total = 0
    for i in range(12):
        total += np.random.uniform(0, 1)
    return total - 6


def update(new_val, mean, var, count, var_temp):
    count += 1
    new_mean = mean + (new_val - mean)/count
    if count != 1:
        new_var_temp = var_temp + (new_val - mean)*(new_val - new_mean)
        new_var = new_var_temp/(count-1)
    else:
        new_var, new_var_temp = 0, 0
    return new_mean, new_var, count, new_var_temp


if __name__ == "__main__":

    # input / create dataset
    mean = float(input())
    var = float(input())
    print("Data point source function: N(" + str(mean) + ", " + str(var) + ")")
    print()
    sample_mean, sample_var, count, var_temp = 0, 0, 0, 0
    # maximum trial 10000
    epoch = 10000
    while(epoch):
        data = univariate_generator(mean, var)
        print("Add data point: " + str(data))
        new_sample_mean, new_sample_var, count, new_var_temp = update(
            data, sample_mean, sample_var, count, var_temp)
        print("Mean = " + str(new_sample_mean) +
              "   Variance = " + str(new_sample_var))

        if(abs(new_sample_mean - sample_mean) < 0.001 and abs(new_sample_var - sample_var) < 0.001):
            break
        else:
            sample_mean = new_sample_mean
            sample_var = new_sample_var
            var_temp = new_var_temp
            epoch -= 1
