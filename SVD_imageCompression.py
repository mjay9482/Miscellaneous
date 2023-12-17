#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:20:36 2023

@author: Mjay
"""
# # SVD image compression


import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np

plt.rcParams['figure.figsize'] = [16, 8]

image_path = 'parrot.jpg'
original_image = imread(image_path)
gray_image = np.mean(original_image, -1)

original_plot = plt.imshow(gray_image)
original_plot.set_cmap('gray')
plt.axis('off')
plt.show()

U, S, VT = np.linalg.svd(gray_image, full_matrices=False)
S = np.diag(S)

subplot_index = 0
for rank in (5, 20, 100):
    # Construct an Approximate image
    approx_image = U[:, :rank] @ S[0:rank, :rank] @ VT[:rank, :]
    plt.figure(subplot_index + 1)
    subplot_index += 1
    approx_plot = plt.imshow(approx_image)
    approx_plot.set_cmap('gray')
    plt.axis('off')
    plt.title(f'Rank = {rank}')
    plt.show()

plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(S)) / np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sum')
plt.show()
