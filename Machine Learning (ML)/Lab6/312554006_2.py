# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:37:53 2023

@author: max2306
"""
import imageio as iio
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

color_dict = {
    0: [0, 0, 150],
    1: [0, 0, 0],
    2: [0, 150, 0],
    3: [150, 0, 0],
    4: [200, 150, 50]
}


def read_file(file_name):
    return iio.imread(file_name)


def kernel(image, gamma_s, gamma_c):

    # spatial distance
    idx = np.zeros((10000, 2))
    for i in range(10000):
        idx[i][0] = i/100
        idx[i][1] = i % 100
    spa_dis = cdist(idx, idx, 'sqeuclidean')

    # color distance
    image_reshape = image.reshape((10000, 3))
    col_dis = cdist(image_reshape, image_reshape, 'sqeuclidean')

    return np.multiply(np.exp(-gamma_s*spa_dis), np.exp(-gamma_c*col_dis))


def normalize(W, cut):
    # based on the definition: L = D - W, and D is diagnoal degree matrix
    D = np.zeros_like(W)
    for i in range(len(D)):
        D[i][i] = np.sum(W[i])
    L = D - W

    # based on different cut method, we have different nomralized way
    # normalize cut

    if cut == 0:
        for i in range(10000):
            D[i][i] = 1 / np.sqrt(D[i][i])
        L_norm = D@L@D
        return L_norm

    # ratio cut
    else:
        L_norm = L
        return L_norm


def compute_U(L_norm, k):
    e_val, e_vec = np.linalg.eig(L_norm)

    non_zero_indices = np.where(np.abs(e_val) > 1e-10)[0]
    e_val = e_val[non_zero_indices]
    e_vec = e_vec[:, non_zero_indices]

    sorted_indices = np.argsort(e_val)
    e_val = e_val[sorted_indices]
    e_vec = e_vec[:, sorted_indices]

    return e_vec[:, :k]


def normalize_U(U):
    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    return U / row_norms


def initial_cluster(method, k, U_norm):

    # original method: to divide different parts depend on positions
    cluster_label = np.zeros(10000)
    if method == 0:
        group = np.random.choice(10000, 2, replace=False)
        for i in range(10000):
            dis = np.zeros(k)
            for j in range(k):
                dis[j] = cdist(U_norm[i].reshape(1, -1),
                               U_norm[group[j]].reshape(1, -1), 'euclidean')
            cluster_label[i] = np.argmin(dis)
        return cluster_label

    else:
        idx = np.zeros((10000, 2))
        for i in range(10000):
            idx[i][0] = int(i//100)
            idx[i][1] = int(i % 100)
        center = idx[np.random.choice(len(idx))]
        center = [[int(x) for x in np.array(center)]]

        for i in range(1, k):
            dis = np.array(
                [min([np.linalg.norm(x - c)**2 for c in center]) for x in idx])
            prob = dis / dis.sum()
            new_center = idx[np.random.choice(len(idx), p=prob)]
            new_center = [int(x) for x in np.array(new_center)]
            center.append(new_center)

        cluster_label = np.ones(10000)
        for i in range(10000):
            dis = np.zeros(k)
            for j in range(k):
                pos = center[j][0]*100 + center[j][1]

                dis[j] = cdist(U_norm[i].reshape(1, -1),
                               U_norm[pos].reshape(1, -1), 'euclidean')
            cluster_label[i] = np.argmin(dis)
        return cluster_label


def transfer_img(token_list, image):
    img_list = image.reshape((10000, 3))
    for i in range(10000):
        img_list[i][:] = color_dict[token_list[i]]
    img_list = img_list.reshape((100, 100, 3))

    return Image.fromarray(img_list)


def clustering(U_norm, k, cut, image):

    # initilaization
    token_list = initial_cluster(method, k, U_norm)
    img_res = []
    img_res.append(transfer_img(token_list, image))
    # k-means, iteration = 100
    for i in range(100):
        print("epoch" + str(i))
        token_list_pre = token_list.copy()
        # M step : recalculate the mle
        group = []
        __, count = np.unique(token_list_pre, return_counts=True)
        for j in range(k):
            idx = np.where(token_list_pre == j)
            group.append(np.sum(U_norm[idx], axis=0)/count[j])

        # E-step: assign the group again
        for j in range(10000):
            dis = np.zeros(k)
            for n in range(k):
                dis[n] = cdist(U_norm[j].reshape(1, -1),
                               group[n].reshape(1, -1), 'euclidean')
            token_list[j] = np.argmin(dis)

        img_res.append(transfer_img(token_list, image))
        if (np.sum(token_list != token_list_pre) < 2):
            break

    img_res[0].save(f"{cut}_Spec_before.png")
    img_res[-1].save(f"{cut}_Spec_after.png")
    img_res[0].save(f"{cut}Spec.gif", save_all=True,
                    append_images=img_res[1:], duration=100, loop=0)

    if k == 2:
        # plot eigenspace
        color = ['black', 'blue']
        for i in range(10000):
            plt.scatter(U_norm[i][0], U_norm[i][1],
                        c=color[int(token_list[i])])
        plt.title("the eigenspace")
        plt.show()


if __name__ == "__main__":

    # read file
    img1 = read_file("image1.png")
    img2 = read_file("image2.png")
    image = img1
    gamma_s = 0.001
    gamma_c = 0.01
    cut = 0
    k = 3
    method = 1

    # first step: construct similarity graph (i.e the kernel)
    W = kernel(image, gamma_s, gamma_c)

    # second step : normalized laplacian
    L_norm = normalize(W, cut)

    # third part : find eigenvector and the matrix U
    U = compute_U(L_norm, k)

    # forth part : normalization to 1
    U_norm = normalize_U(U)

    # fifth part: clustering
    clustering(U_norm, k, cut, image)
