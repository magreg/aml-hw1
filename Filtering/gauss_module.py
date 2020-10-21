# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2



"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):
    x = np.arange(-3 * sigma, 3 * sigma + 1)
    Gx = (1 / (np.sqrt(2 * np.pi) * sigma)) * (np.exp(-np.power(x, 2.) / (2 * np.power(sigma, 2.))))    
    return Gx, x



"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""

def conv2d(img, kernel):
    # Flip the kernel
    kernel = np.flip(kernel)

    # Compute the padding value
    F = kernel.shape[0]
    padding = int(np.ceil((F - 1) / 2))

    # Add zero padding to the input img
    img_padded = np.zeros((img.shape[0] + 2 * padding, img.shape[1] + 2 * padding))
    img_padded[padding:-padding, padding:-padding] = img

    # Compute the sub matrices
    sub_matrices = view_as_windows(img_padded, (F, F), 1)
    # Compute the convolution with the element-wise multiplication of every sub matrix and sum
    output = np.einsum('ij, klij -> kl', kernel, sub_matrices)

    return output


def gaussianfilter(img, sigma):
    
    #...

    return smooth_img



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):

    #...
    
    return Dx, x



def gaussderiv(img, sigma):

    #...
    
    return imgDx, imgDy

