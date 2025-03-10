# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:21:04 2023

@author: Max Lo

school id : 312554006

department : Computer Science

Institute : Data Science and Engineering
"""


import numpy as np
import matplotlib.pyplot as plt

# matrix operation
def transpose(matrix):
    row_size = matrix.shape[0]
    col_size = matrix.shape[1]
    matrixT = np.zeros((col_size, row_size))
    for i in range(row_size):
        for j in range(col_size):
            matrixT[j, i] = matrix[i,j]
    return matrixT

def mul(matrixA, matrixB):
    row = matrixA.shape[0]
    col = matrixB.shape[1]
    temp = matrixA.shape[1]
    matrix = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            for k in range(temp):
                matrix[i,j] += matrixA[i,k]*matrixB[k,j]
    return matrix

def matrix_Lam(lam, size):
    matrix = np.zeros((size,size))
    for i in range(size):
        matrix[i,i] = lam 
    return matrix

def LU(matrix):    
    row_size = matrix.shape[0]
    col_size = matrix.shape[1]
    matrix_L = matrix_Lam(1, row_size)
    matrix_U = matrix.copy()

    for j in range(col_size):
        for i in range(j+1,row_size):
            param = -matrix_U[i][j]/matrix_U[j][j]
            for k in range(j, col_size):
                matrix_U[i][k] += param*matrix_U[j][k]
            matrix_L[i][j] = -param

    return matrix_L, matrix_U

def inverse_L(matrix_L):
    # matrix L is unitriangular
    matrix_invL = matrix_L.copy()
    row_size = matrix_L.shape[0]
    col_size = matrix_L.shape[1]
    for i in range(row_size):
        for j in range(col_size):
            if i != j:
                matrix_invL[i][j] *= -1

    return matrix_invL

def inverse_U(matrix_U):
    # matrix U is not unittriangular
    matrix_invU = matrix_U.copy()
    row_size = matrix_U.shape[0]
    col_size = matrix_U.shape[1]
    for j in range(col_size-1, -1, -1):
        # adjust the row
        if matrix_invU[j][j] != 1 :
            param = 1/matrix_invU[j][j]
            for k in range(0, col_size):
                matrix_invU[j][k] *= param
            matrix_invU[j][j] *= param
        # adjust the col
        for i in range(j-1, -1, -1):
            param = -matrix_invU[i][j]
            for k in range(0,col_size):
                matrix_invU[i][k] += param*matrix_invU[j][k]
            matrix_invU[i][j] = param*matrix_invU[j][j]
    return matrix_invU

def inverse(matrix):
    # A = LU
    L, U = LU(matrix)
    # since = A^-1 = U^-1 * L^-1
    invU = inverse_U(U)
    invL = inverse_L(L)
    invA = mul(invU, invL)
    
    return invA

# read file
def read_file(file_name, base):          
    file = open(file_name, "r")
    A = []
    b = []
    xaxis = []
    for line in file:
        listx = []
        x, y = line.split(",")
        x, y = float(x), float(y)
        for i in range(base-1, -1, -1):
            listx.append(pow(x,i))
        A.append(listx)
        b.append(y)
        xaxis.append(x)
    A = np.array(A)
    b = np.array(b).reshape(-1,1)
    xaxis = np.array(xaxis).reshape(-1,1)
    
    return A, b, xaxis

# method : 1. LSE / 2. Steepest gradient decent / 3. Newton's method

def LSE(A, b, lam, size):
    AT = transpose(A)
    inv = inverse(mul(AT, A) + matrix_Lam(lam, size))
    temp = mul(inv, AT)
    return mul(temp, b)


def steepest(A, b, lam, base, xaxis):
    degree = base
    x_pred = np.zeros((degree, 1))
    L = 0.001
    epochs = 10000
    n = float(len(xaxis))
    
    for i in range(epochs):
        Y_pred = mul(A, x_pred)
        for j in range(degree-1, -1, -1):
            temp = (-2/n)*sum(mul(transpose(b-Y_pred),pow(xaxis, j))) + (lam/n)*np.sign(x_pred[degree-1-j][0])
            x_pred[degree-1-j][0] -= L*temp

        Y_pred = mul(A, x_pred)
    return x_pred
    
def Newton(A, b, lam, base, xaxis):
    degree = base
    x_pre = np.zeros((degree, 1))
    epochs = 100000
    for i in range(epochs):
        
        first_d = 2*mul(mul(transpose(A),A), x_pre) - 2*mul(transpose(A), b) 
        second_d = 2*mul(transpose(A), A) 
        x_new = x_pre - mul(inverse(second_d), first_d)
        if(sum(abs(x_new-x_pre)) < 0.001):
            break

        x_pre = x_new
        
    return x_new

# print out the result
def result(method_name, xaxis, A, x, b):
    # print curve
    plt.plot(xaxis, mul(A,x), 'b-', label = "best fitted line")
    plt.plot(xaxis, b, 'ro', label = "data points")
    plt.title("method_name: " + method_name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    
    # print result
    error = mul(A,x)-b
    error_suqare = sum(pow(error,2))
    print(method_name + ":")
    print("Fitting line: ", end =' ')
    for i in range(x.shape[0]-1, -1, -1):
        if np.sign(x[x.shape[0]-1-i][0]) > 0:
            sign = '+ '
        else :
            sign = "- "
        # first term
        if i == x.shape[0]-1 :
            print(str(x[x.shape[0]-1-i][0])+"X^"+str(i), end ="")
        #middle term             
        elif i != 0:
            print(" " + sign + str(x[x.shape[0]-1-i][0])+"X^"+str(i), end="")
        #last term
        else:
            print(" " + sign + str(x[x.shape[0]-1-i][0]), end="\n")
    print("Total error: " + str(error_suqare[0]), end = '\n\n')        




if __name__ == "__main__":
    #file_name = input("file name : ")
    #base = int(input("base: "))
    #lam = int(input("lambda: "))
    file_name = "testfile.txt"
    base = 3
    lam = 10000
    A, b, xaxis = read_file(file_name, base)
    
#----------------------1. LSE---------------------------#
    
    x = LSE(A, b, lam, base)
    result("LSE", xaxis, A, x, b)
    
#-----------------2. steepest gradient------------------#    
    
    x = steepest(A, b, lam, base, xaxis)
    result("steepest descent method", xaxis, A, x, b)

#-------------------------------------------------------#
    
    x = Newton(A, b, lam, base, xaxis)
    result("Newton's method", xaxis, A, x, b)
