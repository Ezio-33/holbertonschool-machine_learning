#!/usr/bin/env python3
"""Implémentation de la propagation avant pour une couche de pooling"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Effectue la propagation avant pour une couche de pooling

    Args:
        A_prev: Entrée de la couche (m, h_prev, w_prev, c_prev)
        kernel_shape: Dimensions du kernel (kh, kw)
        stride: Pas de déplacement (sh, sw)
        mode: 'max' ou 'avg'

    Returns:
        Sortie de la couche de pooling
    """

    # Extraction des dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calcul des dimensions de sortie
    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1

    # Initialisation de la sortie
    output = np.zeros((m, h_out, w_out, c_prev))

    # Parcours de chaque position de sortie
    for i in range(h_out):
        for j in range(w_out):
            # Délimitation de la fenêtre
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            # Extraction de la région d'intérêt
            window = A_prev[:, vert_start:vert_end, horiz_start:horiz_end, :]

            # Application du pooling
            if mode == 'max':
                output[:, i, j, :] = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(window, axis=(1, 2))

    return output
