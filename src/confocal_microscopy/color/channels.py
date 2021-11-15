"""Used for visualisation.
"""
import numpy as np


def red(image):
    zeros = np.zeros_like(image)
    return np.stack([image, zeros, zeros], axis=-1)

def green(image):
    zeros = np.zeros_like(image)
    return np.stack([zeros, image, zeros], axis=-1)

def blue(image):
    zeros = np.zeros_like(image)
    return np.stack([zeros, zeros, image], axis=-1)