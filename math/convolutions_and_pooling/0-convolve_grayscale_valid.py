#!/usr/bin/env python3
"""Module de convolution valide pour images en niveaux de gris"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Effectue une convolution valide sur un batch d'images

    Args:
        images : ndarray shape (m, h, w)
        kernel : ndarray shape (kh, kw)

    Returns:
        ndarray shape (m, h_out, w_out) des images convoluées
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calcul des dimensions de sortie
    h_out = h - kh + 1
    w_out = w - kw + 1

    # Initialisation du résultat
    output = np.zeros((m, h_out, w_out))

    # Double boucle sur les dimensions du kernel
    for i in range(kh):
        for j in range(kw):
            # Extraction de la fenêtre correspondante
            window = images[:, i:i + h_out, j:j + w_out]
            # Application du coefficient du kernel
            output += window * kernel[i, j]

    return output
