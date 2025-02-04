#!/usr/bin/env python3
"""Module de convolution pour images multicanal"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Effectue une convolution sur des images avec canaux

    Args:
        images : ndarray (m, h, w, c)
        kernel : ndarray (kh, kw, c)
        padding : 'same', 'valid' ou tuple (ph, pw)
        stride : tuple (sh, sw)

    Returns:
        ndarray (m, h_out, w_out)
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    # Vérification compatibilité canaux
    if kc != c:
        raise ValueError("Kernel channels must match image channels")

    # Gestion du padding
    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Application du padding sur hauteur/largeur uniquement
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    # Calcul des dimensions de sortie
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, h_out, w_out))

    # Parcours des positions de sortie avec stride
    for i in range(h_out):
        for j in range(w_out):
            # Calcul des positions de la fenêtre
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            # Extraction fenêtre (m, kh, kw, c)
            window = padded[:, h_start:h_end, w_start:w_end, :]

            # Produit élémentaire et somme sur les 3 dimensions
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2, 3))

    return output
