#!/usr/bin/env python3
"""Module de convolution avec stride et padding variable"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Effectue une convolution sur des images en niveaux de gris avec
    gestion avancée

    Args:
        images : ndarray (m, h, w)
        kernel : ndarray (kh, kw)
        padding : tuple ou 'same'/'valid'
        stride : tuple (sh, sw)

    Returns:
        ndarray des images convoluées
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Gestion du padding
    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Application du padding
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Calcul des dimensions de sortie
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, h_out, w_out))

    # Parcours avec stride
    for i in range(h_out):
        for j in range(w_out):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            window = padded[:, vert_start:vert_end, horiz_start:horiz_end]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
