#!/usr/bin/env python3
"""Convolution avec gestion correcte du stride et padding 'same'"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Effectue une convolution sur des images en niveaux de gris

    Args:
        images : ndarray (m, h, w)
        kernel : ndarray (kh, kw)
        padding : 'same'/'valid'/tuple
        stride : tuple (sh, sw)

    Returns:
        ndarray des images convoluées
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Correction du padding pour 'same' avec stride
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Application du padding
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Nouveau calcul des dimensions
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, h_out, w_out))

    # Parcours corrigé avec gestion stride/kernel
    for i in range(h_out):
        for j in range(w_out):
            x_start = i * sh
            x_end = x_start + kh
            y_start = j * sw
            y_end = y_start + kw

            window = padded[:, x_start:x_end, y_start:y_end]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
