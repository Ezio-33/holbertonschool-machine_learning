#!/usr/bin/env python3
"""Module de convolution avec padding personnalisé"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Effectue une convolution avec padding personnalisé sur des images
    en niveaux de gris

    Args:
        images : ndarray shape (m, h, w)
        kernel : ndarray shape (kh, kw)
        padding : tuple (ph, pw) - padding à appliquer

    Returns:
        ndarray shape (m, h_out, w_out)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Application du padding
    padded_images = np.pad(images,
                           ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    # Calcul des dimensions de sortie
    h_out = h + 2 * ph - kh + 1
    w_out = w + 2 * pw - kw + 1

    output = np.zeros((m, h_out, w_out))

    for i in range(kh):
        for j in range(kw):
            output += padded_images[:, i:i + h_out, j:j + w_out] * kernel[i, j]

    return output
