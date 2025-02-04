#!/usr/bin/env python3
"""Module de pooling (max/avg) pour images multicanal"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Effectue un pooling sur des images

    Args:
        images : ndarray (m, h, w, c)
        kernel_shape : (kh, kw)
        stride : (sh, sw)
        mode : 'max' ou 'avg'

    Returns:
        ndarray (m, h_out, w_out, c)
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calcul dimensions de sortie
    h_out = (h - kh) // sh + 1
    w_out = (w - kw) // sw + 1

    output = np.zeros((m, h_out, w_out, c))

    for i in range(h_out):
        for j in range(w_out):
            # Position de la fenêtre
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            # Extraction fenêtre
            window = images[:, h_start:h_end, w_start:w_end, :]

            # Application pooling
            if mode == 'max':
                output[:, i, j, :] = np.max(window, axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(window, axis=(1, 2))

    return output
