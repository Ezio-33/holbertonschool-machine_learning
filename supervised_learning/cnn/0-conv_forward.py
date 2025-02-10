#!/usr/bin/env python3
"""Propagation avant pour une couche convolutive"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Effectue la propagation avant d'une couche convolutive.

    Args:
        A_prev: Entrée de forme (m, h_prev, w_prev, c_prev)
        W: Filtres (kh, kw, c_prev, c_new)
        b: Biais (1, 1, 1, c_new)
        activation: Fonction d'activation
        padding: 'same' ou 'valid'
        stride: Pas (sh, sw)

    Returns:
        Sortie activée de la convolution
    """

    # Dimensions d'entrée
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Calcul du padding selon le type
    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:  # 'valid'
        ph, pw = 0, 0

    # Application du padding
    A_pad = np.pad(A_prev,
                   ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                   mode='constant')

    # Dimensions de sortie
    h_out = (h_prev + 2 * ph - kh) // sh + 1
    w_out = (w_prev + 2 * pw - kw) // sw + 1

    # Initialisation de la sortie
    Z = np.zeros((m, h_out, w_out, c_new))

    # Parcours de chaque position de sortie
    for i in range(h_out):
        for j in range(w_out):
            # Calcul de la fenêtre
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            # Extraction de la région
            a_slice = A_pad[:, vert_start:vert_end, horiz_start:horiz_end, :]

            # Calcul convolution + biais pour tous les filtres
            for c in range(c_new):
                Z[:, i, j, c] = np.sum(
                    a_slice * W[..., c], axis=(1, 2, 3)) + b[..., c]

    return activation(Z)
