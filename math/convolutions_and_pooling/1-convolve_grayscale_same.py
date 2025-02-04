#!/usr/bin/env python3
"""Module de convolution same pour images en niveaux de gris"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Effectue une convolution same sur un batch d'images

    Args:
        images : ndarray shape (m, h, w)
        kernel : ndarray shape (kh, kw)

    Returns:
        ndarray shape (m, h, w) des images convoluées
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calcul du padding
    ph = (kh - 1) // 2
    pw = (kw - 1) // 2

    # Application du padding
    padded_images = np.pad(images,
                           ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    # Initialisation sortie même dimension qu'entrée
    output = np.zeros((m, h, w))

    # Convolution valide sur l'image padée
    for i in range(h):
        for j in range(w):
            # Extraction de la fenêtre correspondante
            window = padded_images[:, i:i + kh, j:j + kw]
            # Application du kernel et somme
            result = np.sum(window * kernel[np.newaxis, :, :], axis=(1, 2))
            output[:, i, j] = result

    return output
