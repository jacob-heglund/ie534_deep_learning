import numpy as np
import h5py
import time
import copy
from random import randint
import matplotlib.pyplot as plt
from scipy import signal

x = np.ones([28,28])
K = np.ones([26,26])
# this only works for "filter" size 26x26 (for backprop)
def myConvolve2D(x, K):
    imageSide = np.shape(x)[0]

    kx = np.shape(K)[0]
    ky = np.shape(K)[1]
    print(kx, ky)
    outputMatrixSizeX = imageSide - kx + 1
    outputMatrixSizeY = imageSide - ky + 1

    Z = np.zeros([outputMatrixSizeX, outputMatrixSizeY])

    for i in range(outputMatrixSizeX):
        for j in range(outputMatrixSizeY):
            # TODO: this will change depending on filter size, but works for a 26x26
            imageSlice = x[i:i+kx, j:j+ky]
            # do elementwise multiplication
            intermediateMatrix = np.multiply(imageSlice, K)
            total = np.sum(intermediateMatrix) 
            Z[i, j] = total
            
    return Z
a = myConvolve2D(x, K)
print(np.shape(a))

