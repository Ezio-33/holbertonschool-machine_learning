#!/usr/bin/env python3
"""Module pour la rétropropagation d'une couche convolutive"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Calcule la rétropropagation pour une couche convolutive

    Args:
        dZ: Gradients de sortie (m, h_new, w_new, c_new)
        A_prev: Entrée précédente (m, h_prev, w_prev, c_prev)
        W: Filtres (kh, kw, c_prev, c_new)
        b: Biais (1, 1, 1, c_new)
        padding: Type de padding ('same' ou 'valid')
        stride: Pas de la convolution (sh, sw)

    Returns:
        dA_prev, dW, db: Gradients pour l'entrée, les filtres et les biais
    """
    # Récupération des dimensions
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    # Initialisation du padding
    if padding == 'valid':
        ph = pw = 0
    else:  # same
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1

    # Application du padding à A_prev
    A_pad = np.pad(A_prev,
                   ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                   mode='constant')

    # Initialisation des gradients
    dW = np.zeros_like(W)
    dA_prev = np.zeros_like(A_pad)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Calcul des gradients
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    # Extraction des portions nécessaires
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    # Calcul des gradients
                    aux_W = W[:, :, :, c]
                    aux_dZ = dZ[i, h, w, c]

                    # Mise à jour des gradients
                    dA_prev[i, vert_start:vert_end,
                            horiz_start:horiz_end] += aux_W * aux_dZ
                    dW[:, :, :, c] += A_pad[i, vert_start:vert_end,
                                            horiz_start:horiz_end] * aux_dZ

    # Retrait du padding
    if padding == "same":
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
