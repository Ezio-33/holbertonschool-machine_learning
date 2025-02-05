#!/usr/bin/env python3
"""Convolution multi-kernels avec gestion avancée du padding et stride"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Effectue une convolution avec plusieurs kernels

    Args:
        images : ndarray (m, h, w, c)
        kernels : ndarray (kh, kw, c, nc)
        padding : 'same'/'valid'/tuple
        stride : (sh, sw)

    Returns:
        ndarray (m, h_out, w_out, nc)
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    # Vérification des canaux
    if kc != c:
        raise ValueError("Les canaux du kernel doivent correspondre à l'image")

    # Calcul du padding avec gestion du stride
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Application du padding
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    # Calcul des dimensions de sortie
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, h_out, w_out, nc))

    # Parcours spatial optimisé
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            # Extraction fenêtre vectorisée
            window = padded[:, h_start:h_end, w_start:w_end, :]

            # Calcul simultané pour tous les kernels
            output[:, i, j, :] = np.tensordot(
                window, kernels, axes=([1, 2, 3], [0, 1, 2]))

    return output
