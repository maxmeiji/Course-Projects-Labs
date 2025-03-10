# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os
import re
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

def RBF(X_1, X_2, gamma):
    dis = np.sum(X_1**2, axis = 1).reshape(-1,1) + np.sum(X_2**2, axis = 1) - 2*X_1@X_2.T
    rbf = np.exp(-gamma*dis)
    return rbf

def poly(X_1, X_2, c, degree):
    return np.power((X_1@X_2.T+c), degree)

def linear(X_1, X_2):
    return X_1@X_2.T


def read_file(file_name):

    img = []
    label = []
    for file in os.listdir(file_name):
        path = os.path.join(file_name, file)
        img_array = Image.open(path)
        img_array = img_array.resize((50,50))

        img.append(np.array(img_array).ravel())
        label.append(int(re.search(r'(\d+)', file).group(1)))
    
    return np.array(img), label

def PCA(data, dimension):
    
    mean = np.mean(data, axis = 0)
    centered_data = data - mean
    
    # the cov of data
    S = centered_data.T @ centered_data /(data.shape[0] - 1)
    
    # the first k eigenvalues and their corresponding eigenvectors
    eig_val, eig_vec = np.linalg.eigh(S)
    # normalize to 1
    eig_vec = eig_vec / np.linalg.norm(eig_vec, axis=0)
    
    sort_idx = np.argsort(eig_val)[::-1]
    eig_vec = eig_vec[:,sort_idx]
    
    W = eig_vec[:, :dimension]

    return W

def kernel_PCA(data, dimension, method):
    
    mean = np.mean(data, axis = 0)
    centered_data = data - mean
    if method == 1: 
        kernel = RBF(centered_data, centered_data, 0.0000001)
    elif method == 2:
        kernel = poly(centered_data,centered_data, 10, 3)
    else:
        kernel = linear(centered_data, centered_data)

    eig_val, eig_vec = np.linalg.eigh(kernel)
    
    eig_vec = eig_vec / np.linalg.norm(eig_vec, axis=0)
    
    sort_idx = np.argsort(eig_val)[::-1]
    eig_vec = eig_vec[:,sort_idx]
    
    W = eig_vec[:, :dimension]   

    return W
    
    
    
    
    
def LDA(data, label, dimension):
    cat, count = np.unique(label, return_counts=True)    
    size = data.shape[1]
    Sw = np.zeros((size, size))
    Sb = np.zeros((size, size))
    mean_all = np.mean(data, axis = 0)
    for i in range(len(cat)):
        inclass_data = np.array([data[j] for j in range(len(label)) if label[j] == cat[i]])
        inclass_mean = np.mean(inclass_data, axis = 0)
        Sw += (inclass_data-inclass_mean).T @ (inclass_data-inclass_mean)
        Sb += count[i]*((inclass_mean - mean_all).T@(inclass_mean-mean_all))
    
    W = np.linalg.pinv(Sw)@Sb
    eig_val, eig_vec = np.linalg.eigh(W)
    eig_vec = eig_vec / np.linalg.norm(eig_vec, axis=0)
    
    sort_idx = np.argsort(eig_val)[::-1]
    eig_vec = eig_vec[:,sort_idx]
    
    W = eig_vec[:, :dimension]
    
    return W

def kernel_LDA(data, label, dimension, method):
    cat, count = np.unique(label, return_counts=True)    
    L = np.ones((data.shape[0], data.shape[0]))/count[0]
    mean = np.mean(data, axis = 0)
    centered_data = data - mean
    if method == 1: 
        kernel = RBF(centered_data, centered_data, 0.0000001)
    elif method == 2:
        kernel = poly(centered_data,centered_data, 10, 3)
    else:
        kernel = linear(centered_data, centered_data)
    Sw = kernel@kernel
    Sb = kernel@L@kernel
    W = np.linalg.pinv(Sw)@Sb
    eig_val, eig_vec = np.linalg.eigh(W)
    eig_vec = eig_vec / np.linalg.norm(eig_vec, axis=0)
    
    sort_idx = np.argsort(eig_val)[::-1]
    eig_vec = eig_vec[:,sort_idx]
    
    W = eig_vec[:, :dimension]
    return W
        
def result(W, data):
    # projection
    fig, axes = plt.subplots(5, 5, figsize=(12, 12), subplot_kw={'xticks': [], 'yticks': []})    
    for i in range(25):
        row = i // 5
        col = i % 5
        img = W[:, i].reshape(50,50)
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'Eigenface {i + 1}')
    plt.show()
    
    # reconstruction
    idx = np.random.choice(np.arange(data.shape[0]), size=10, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(10, 6), subplot_kw={'xticks': [], 'yticks': []})    
    for i in range(10):
        row = i // 5
        col = i % 5
        img = data[idx[i]]
        re = img@W@W.T
        axes[row, col].imshow(re.reshape(50,50), cmap='gray')
        axes[row, col].set_title(f'reconstruction {i + 1}')
    plt.show()
    

def predict(W, train, test, label_tr, label_tt, k, method):
    if method == 0: 
        proj_tr = train@W
        proj_test = test@W
    else:
        mean_tr = np.mean(train, axis = 0)
        train = train - mean_tr
        mean_tt = np.mean(test, axis = 0)
        test = test - mean_tt        
        if method == 1:
            kernel_tr = RBF(train, train, 0.0000001)
            kernel_tt = RBF(test, train, 0.0000001)
        elif method == 2:
            kernel_tr = poly(train,train, 10, 3)
            kernel_tt = poly(test, train, 10, 3)
        else:
            kernel_tr = linear(train, train)
            kernel_tt = linear(test, train)
        proj_tr = kernel_tr@W
        proj_test = kernel_tt@W
    
    wrong = 0
    
    for i in range(len(proj_test)):
        dis = np.zeros(len(proj_tr))
        pred_label = np.zeros(16)
       
        for j in range(len(proj_tr)):
            dis[j] = euclidean(proj_test[i], proj_tr[j])
        sort_idx = np.argsort(dis)[:k]
        for j in range(k):
            pred_label[label_tr[sort_idx[j]]] += 1
        ans = np.argmax(pred_label)
        #print(ans, label_tt[i])
        if ans != label_tt[i]:
            wrong += 1
    
    acc = float(len(label_tt)-wrong) / (len(label_tt))
    
    return acc
    

                
                 

if __name__ == "__main__":
    
    # read file    
    file_name = os.getcwd()
    tr_data_dir = os.path.join(file_name, 'Yale_Face_Database\\training')
    tt_data_dir = os.path.join(file_name, 'Yale_Face_Database\\testing')
    
    X_tr, label_tr = read_file(tr_data_dir)
    X_tt, label_tt = read_file(tt_data_dir)
    '''
    # task 1: the PCA and LDA eigenspaces and reconstruction   
    W_pca = PCA(X_tr, 25)
    result(W_pca, X_tr)
    
    W_lda = LDA(X_tr, label_tr, 25)
    result(W_lda, X_tr)
    
    # task 2 face recognization
    acc_pca = predict(W_pca, X_tr, X_tt, label_tr, label_tt, 5, 0)
    print("PCA accuracy: ", f"{acc_pca*100:.2f}", "%")
    
    acc_lda = predict(W_lda, X_tr, X_tt, label_tr, label_tt, 5, 0)    
    print("LDA accuracy: ", f"{acc_lda*100:.2f}", "%")
'''
        
    # task 3 : PCA different kernel
    method =3
    kernelP_W = kernel_PCA(X_tr, 25, method)
    acc_kernelp = predict(kernelP_W, X_tr, X_tt, label_tr, label_tt, 5, method)
    print("PCA kernel accuracy: ", f"{acc_kernelp*100:.2f}", "%")
    
    # task 3: LDA different kernel
    # method = 1
    kernelL_W = kernel_LDA(X_tr, label_tr, 25, method)
    acc_kernelp = predict(kernelL_W, X_tr, X_tt, label_tr, label_tt, 5, method)
    print("LDA kernel accuracy: ", f"{acc_kernelp*100:.2f}", "%")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    