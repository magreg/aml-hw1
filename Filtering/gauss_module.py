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
from skimage.util.shape import view_as_windows
from enum import Enum

class ConvolutionType(Enum):
    TWO_D = 1
    VERTICAL = 2
    HORIZONTAL = 3

# Vectorize the convolution code
def conv(img, kernel, conv_type=ConvolutionType.TWO_D):
    # Flip the kernel
    kernel = np.flip(kernel)

    # Reshape the kernel according to the convolution type
    if conv_type == ConvolutionType.VERTICAL:
        kernel = kernel.reshape(kernel.shape[0], 1)
    elif conv_type == ConvolutionType.HORIZONTAL:
        kernel = kernel.reshape(1, kernel.shape[0])

    # Compute the padding value and add zero padding to the input image
    kern_dim = np.array(kernel.shape)
    padding = np.ceil((kern_dim - 1) / 2).astype(int)
    img_padded = np.zeros((img.shape[0] + 2 * padding[0], img.shape[1] + 2 * padding[1]))

    img_padded[padding[0]:img_padded.shape[0] - padding[0], padding[1]:img_padded.shape[1] - padding[1]] = img

    # Compute the submatrices according to the kernel
    sub_matrices = view_as_windows(img_padded, tuple(kern_dim), 1)

    # Compute the convolution with the element-wise multiplication of every sub matrix and sum
    return np.einsum('ij, klij -> kl', kernel, sub_matrices)


def gaussianfilter(img, sigma):
    # Compute the 1d-Gaussian
    Gx, x = gauss(sigma)

    # Convolute twice with the Gaussian filter, first vertically and then horizontally
    smooth_img = conv(img, Gx, conv_type=ConvolutionType.VERTICAL)
    return conv(smooth_img, Gx, conv_type=ConvolutionType.HORIZONTAL)


"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):
    x = np.arange(-3 * sigma, 3 * sigma + 1)
    Dx = - (x / (np.sqrt(2 * np.pi) * np.power(sigma, 3.))) * (np.exp(-np.power(x, 2.) / (2 * np.power(sigma, 2.))))    
    return Dx, x



def gaussderiv(img, sigma):

    #...
    
    return imgDx, imgDy

