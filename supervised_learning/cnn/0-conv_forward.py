#!/usr/bin/env python3
"""Implémentation de la propagation avant pour une couche convolutive"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Effectue la propagation avant pour une couche convolutive

    Args:
        A_prev: Sortie de la couche précédente (m, h_prev, w_prev, c_prev)
        W: Filtres de convolution (kh, kw, c_prev, c_new)
        b: Biais (1, 1, 1, c_new)
        activation: Fonction d'activation
        padding: 'same' ou 'valid'
        stride: Tuple (sh, sw)

    Returns:
        Sortie de la couche convolutive après activation
    """

    # Extraction des dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Calcul des dimensions de sortie
    if padding == "same":
        h_out = np.ceil(h_prev / sh).astype(int)
        w_out = np.ceil(w_prev / sw).astype(int)

        # Calcul du padding nécessaire
        pad_h = max((h_out - 1) * sh + kh - h_prev, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_w = max((w_out - 1) * sw + kw - w_prev, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Application du padding
        A_pad = np.pad(A_prev,
                       ((0, 0), (pad_top, pad_bottom),
                        (pad_left, pad_right), (0, 0)),
                       mode='constant')
    else:
        h_out = (h_prev - kh) // sh + 1
        w_out = (w_prev - kw) // sw + 1
        A_pad = A_prev

    # Initialisation de la sortie
    Z = np.zeros((m, h_out, w_out, c_new))

    # Calcul de la convolution fenêtre par fenêtre
    for i in range(h_out):
        for j in range(w_out):
            # Position de la fenêtre
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            # Extraction de la fenêtre
            window = A_pad[:, v_start:v_end, h_start:h_end, :]

            # Calcul du produit tensoriel pour tous les exemples et filtres
            Z[:, i, j, :] = np.tensordot(window, W, axes=(
                [1, 2, 3], [0, 1, 2])) + b[0, 0, 0, :]

    return activation(Z)
