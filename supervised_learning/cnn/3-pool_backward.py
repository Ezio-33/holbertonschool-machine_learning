#!/usr/bin/env python3
"""Rétropropagation pour couche de pooling"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Calcule le gradient pour la couche de pooling

    Args:
        dA: Gradient de la sortie (m, h_out, w_out, c)
        A_prev: Entrée originale (m, h_prev, w_prev, c)
        kernel_shape: (kh, kw) dimensions de la fenêtre
        stride: (sh, sw) pas de déplacement
        mode: 'max' ou 'avg'

    Returns:
        dA_prev: Gradient pour l'entrée précédente
    """
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_out = dA.shape[1]
    w_out = dA.shape[2]

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for ch in range(c):
                    # Délimitation de la fenêtre
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    # Extraction de la fenêtre originale
                    window = A_prev[i, vert_start:vert_end,
                                    horiz_start:horiz_end, ch]

                    if mode == 'max':
                        # Création d'un masque binaire pour le max
                        mask = (window == np.max(window))
                        dA_prev[i,
                                vert_start:vert_end,
                                horiz_start:horiz_end,
                                ch] += mask * dA[i,
                                                 h,
                                                 w,
                                                 ch]

                    elif mode == 'avg':
                        # Répartition uniforme du gradient
                        avg_grad = dA[i, h, w, ch] / (kh * kw)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end,
                                ch] += np.ones((kh, kw)) * avg_grad

    return dA_prev
