# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 17:36:46 2023

@author: max2306

ML 2023 Fall, 312554006 Ming-Chih, Lo
"""

# read image package
import imageio as iio
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image, ImageDraw

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
        idx[i][0] = i//100
        idx[i][1] = i % 100
    spa_dis = cdist(idx, idx, 'sqeuclidean')

    # color distance
    image_reshape = image.reshape((10000, 3))
    col_dis = cdist(image_reshape, image_reshape, 'sqeuclidean')

    return np.multiply(np.exp(-gamma_s*spa_dis), np.exp(-gamma_c*col_dis))


def initial_cluster(method, n, kernel_matrix):

    # original method: to divide different parts depend on positions
    if method == 0:
        cluster_label = np.random.randint(0, n, size=10000)
        return cluster_label
    # k-means++
    else:
        idx = np.zeros((10000, 2))
        for i in range(10000):
            idx[i][0] = int(i//100)
            idx[i][1] = int(i % 100)
        center = idx[np.random.choice(len(idx))]
        center = [[int(x) for x in np.array(center)]]

        for i in range(1, n):
            dis = np.array(
                [min([np.linalg.norm(x - c)**2 for c in center]) for x in idx])
            prob = dis / dis.sum()
            new_center = idx[np.random.choice(len(idx), p=prob)]
            new_center = [int(x) for x in np.array(new_center)]
            center.append(new_center)

        cluster_label = np.ones(10000)
        for i in range(10000):

            min_distance = np.inf
            for j in range(n):
                pos = center[j][0]*100 + center[j][1]
                dis = kernel_matrix[i][i] - \
                    (2/1)*kernel_matrix[i][pos] + kernel_matrix[pos][pos]
                print(dis, min_distance)
                if dis < min_distance:

                    min_distance = dis
                    cluster_label[i] = j

        return cluster_label


def kernel_kmeans(image, n, method, gamma_s, gamma_c):

    # initilaization
    kernel_matrix = kernel(image, gamma_s, gamma_c)
    token_list = initial_cluster(method, n, kernel_matrix)

    img_res = []

    for i in range(100):

        token_list_pre = token_list.copy()
        # claculate count of each assigned group
        __, count = np.unique(token_list_pre, return_counts=True)
        print(count)
        # calculate group kernel first
        kernel_pq = np.zeros(n)

        for k in range(n):
            kernel_matrix_pre = kernel_matrix.copy()
            indices = np.where(token_list_pre == k)[0]
            for j in range(10000):
                if token_list_pre[j] != k:
                    kernel_matrix_pre[j, :] = 0
                    kernel_matrix_pre[:, j] = 0
            kernel_pq[k] = np.sum(kernel_matrix_pre)
        # E-step
        for j in range(10000):
            min_distance = np.inf
            for k in range(n):
                # first part : own kernel
                first_part_kernel = kernel_matrix[j][j]

                # second part : own-group kernel
                #our_group = token_list[j]
                indices = np.where(token_list_pre == k)[0]
                second_part_kernel = -2*(
                    2/count[k])*np.sum(kernel_matrix[j][indices])

                # third part : group kernel
                third_part_kernel = (1/np.power(count[k], 2))*kernel_pq[k]

                total_distance = first_part_kernel + second_part_kernel + third_part_kernel
                if total_distance < min_distance:
                    min_distance = total_distance
                    token_list[j] = k
        img_res.append(transfer_img(token_list, image))

        if (np.sum(token_list != token_list_pre) < 15):
            break

    img_res[0].save("Kmeans_before.png")
    img_res[-1].save("Kmeans_after.png")
    img_res[0].save("Kmeans.gif", save_all=True,
                    append_images=img_res[1:], duration=100, loop=0)


def transfer_img(token_list, image):
    img_list = image.reshape((10000, 3))
    for i in range(10000):
        img_list[i][:] = color_dict[token_list[i]]
    img_list = img_list.reshape((100, 100, 3))

    return Image.fromarray(img_list)


if __name__ == "__main__":

    # read file
    img1 = read_file("image1.png")
    img2 = read_file("image2.png")

    # input parameters
    gamma_s = 0.001
    gamma_c = 0.01
    method = 0
    k = 2
    # task 1: kernel clustering
    kernel_kmeans(img2, k, method, gamma_s, gamma_c)
