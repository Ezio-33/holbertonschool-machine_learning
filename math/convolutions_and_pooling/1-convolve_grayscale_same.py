#!/usr/bin/env python3
"""Module de convolution same pour images en niveaux de gris"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Effectue une convolution same sur des images en niveaux de gris

    Args:
        images : ndarray shape (m, h, w) contenant les images
        kernel : ndarray shape (kh, kw) contenant le kernel

    Returns:
        ndarray contenant les images convoluées
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Gestion du padding pour kernels pairs et impairs
    ph = (kh - int(kh % 2 == 1)) // 2
    pw = (kw - int(kw % 2 == 1)) // 2

    # Calcul des dimensions de sortie
    h_out = h
    w_out = w

    # Application du padding
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant',
                    constant_values=0)

    # Initialisation de la sortie
    output = np.zeros((m, h_out, w_out))

    # Convolution
    for i in range(h_out):
        for j in range(w_out):
            # Extraction de la fenêtre correcte
            window = padded[:, i:i + kh, j:j + kw]
            # Multiplication et somme sur les axes corrects
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
