#!/usr/bin/env python3
"""Module de convolution avec multiples kernels"""

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

    # Vérification compatibilité canaux
    if kc != c:
        raise ValueError("Kernel channels must match image channels")

    sh, sw = stride

    # Gestion du padding
    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Application padding (ne pas padder les canaux)
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    # Calcul dimensions de sortie
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1

    # Initialisation sortie avec nc canaux
    output = np.zeros((m, h_out, w_out, nc))

    # Boucle sur chaque kernel (3ème boucle autorisée)
    for k in range(nc):
        kernel = kernels[:, :, :, k]  # Kernel courant

        # Boucles spatiales
        for i in range(h_out):
            for j in range(w_out):
                # Calcul des positions avec stride
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw

                # Extraction fenêtre (m, kh, kw, c)
                window = padded[:, h_start:h_end, w_start:w_end, :]

                # Produit + somme sur kh, kw, c
                output[:, i, j, k] = np.sum(window * kernel, axis=(1, 2, 3))

    return output
