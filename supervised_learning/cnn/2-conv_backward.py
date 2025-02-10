#!/usr/bin/env python3
"""Rétropropagation pour couche convolutive corrigée"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Calcule les gradients pour la couche convolutive

    Args:
        dZ: Gradient de la sortie (m, h_out, w_out, c_new)
        A_prev: Entrée originale (m, h_prev, w_prev, c_prev)
        W: Filtres (kh, kw, c_prev, c_new)
        b: Biais (1, 1, 1, c_new)
        padding: 'same' ou 'valid'
        stride: (sh, sw)

    Returns:
        dA_prev, dW, db - Les gradients à propager
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Initialisation des gradients
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Gestion du padding
    if padding == "same":
        pad_h = ((h_prev - 1) * sh + kh - h_prev) // 2
        pad_w = ((w_prev - 1) * sw + kw - w_prev) // 2
        A_pad = np.pad(
            A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    else:
        pad_h = pad_w = 0
        A_pad = A_prev

    # Calcul des gradients
    for i in range(m):
        for h in range(dZ.shape[1]):
            for w in range(dZ.shape[2]):
                for c in range(c_new):
                    # Extraction de la fenêtre
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    # Mise à jour des gradients
                    window = A_pad[i, vert_start:vert_end,
                                   horiz_start:horiz_end, :]
                    dW[:, :, :, c] += window * dZ[i, h, w, c]
                    dA_prev[i,
                            vert_start:vert_end,
                            horiz_start:horiz_end,
                            :] += W[:,
                                    :,
                                    :,
                                    c] * dZ[i,
                                            h,
                                            w,
                                            c]

    # Retrait du padding si nécessaire
    if padding == "same":
        dA_prev = dA_prev[:, pad_h:h_prev + pad_h, pad_w:w_prev + pad_w, :]

    return dA_prev, dW, db
