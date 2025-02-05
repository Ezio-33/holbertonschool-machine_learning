#!/usr/bin/env python3
"""Convolution multicanal avec gestion correcte du padding et stride"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Effectue une convolution sur des images avec canaux

    Args:
        images : ndarray (m, h, w, c)
        kernel : ndarray (kh, kw, c)
        padding : 'same'/'valid'/tuple
        stride : tuple (sh, sw)

    Returns:
        ndarray (m, h_out, w_out)
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    # Vérification canaux
    if kc != c:
        raise ValueError("Les canaux du kernel doivent correspondre à l'image")

    # Nouveau calcul du padding pour 'same'
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

    # Dimensions de sortie
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, h_out, w_out))

    # Parcours des positions
    for i in range(h_out):
        for j in range(w_out):
            # Positions avec gestion stride
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            # Fenêtre (m, kh, kw, c)
            window = padded[:, h_start:h_end, w_start:w_end, :]

            # Produit + somme sur kh, kw, c
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2, 3))

    return output
