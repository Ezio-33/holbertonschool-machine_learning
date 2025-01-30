#!/usr/bin/env python3
"""
Module implémentant la descente de gradient avec régularisation Dropout
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Met à jour les poids d'un réseau de neurones avec régularisation Dropout

    Args:
        Y: numpy.ndarray, labels one-hot (classes × m)
        weights: dict, poids et biais du réseau
        cache: dict, sorties et masques dropout de chaque couche
        alpha: float, taux d'apprentissage
        keep_prob: float, probabilité de garder un neurone
        L: int, nombre de couches

    Returns:
        None, met à jour les poids in-place
    """
    m = Y.shape[1]
    dZ = cache[f'A{L}'] - Y

    for layer in range(L, 0, -1):
        A_prev = cache[f'A{layer-1}']
        W = weights[f'W{layer}']

        # Calcul des gradients
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if layer > 1:
            # Calcul de dZ pour la couche précédente
            dA = np.matmul(W.T, dZ)
            # Application du masque dropout
            dA = dA * cache[f'D{layer-1}']
            dA = dA / keep_prob
            # Dérivée de tanh
            dZ = dA * (1 - np.square(A_prev))

        # Mise à jour des poids et biais
        weights[f'W{layer}'] -= alpha * dW
        weights[f'b{layer}'] -= alpha * db
