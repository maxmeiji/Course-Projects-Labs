# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 05:35:44 2023

@author: max2306
"""

import numpy as np
import math


def Binominal(num_list, alpha, beta):

    a_org = alpha
    b_org = beta
    for i in range(len(num_list)):
        ones = 0
        zeros = 0
        for j in range(len(num_list[i])):
            if(num_list[i][j] == "0"):
                zeros += 1
            else:
                ones += 1
        likelihood = mle(ones, zeros, len(num_list[i]))
        a_new = a_org + ones
        b_new = b_org + zeros
        result(i, num_list[i], likelihood, a_org, b_org, a_new, b_new)
        a_org = a_new
        b_org = b_new


def result(i, line, likelihood, a_org, b_org, a_new, b_new):
    print("case " + str(i+1) + ": " + line)
    print("Likelihood: " + str(likelihood))
    print("Beta prior: a = " + str(a_org) + " b = " + str(b_org))
    print("Beta posterior: a = " + str(a_new) + " b = " + str(b_new))
    print("\n")


def mle(ones, zeros, num):
    p = ones/num
    return math.factorial(num) / math.factorial(ones) / math.factorial(zeros) \
        * np.power(p, ones) * np.power((1-p), zeros)


def read_file(file_name):
    file = open(file_name, 'r')
    num_list = list(file)
    num_list = list(map(str.strip, num_list))
    return num_list


if __name__ == "__main__":
    file_name = "testfile.txt"
    num_list = read_file(file_name)
    Binominal(num_list, 10, 1)
