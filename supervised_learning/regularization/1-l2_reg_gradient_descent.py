#!/usr/bin/env python3
"""
Module implémentant la descente de gradient avec régularisation L2
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Met à jour les poids et biais d'un réseau de neurones en utilisant
    la descente de gradient avec régularisation L2.

    Parameters:
        Y (numpy.ndarray): Labels one-hot encodés (classes × m)
        weights (dict): Poids et biais du réseau
        cache (dict): Sorties de chaque couche
        alpha (float): Taux d'apprentissage
        lambtha (float): Paramètre de régularisation L2
        L (int): Nombre de couches

    Returns:
        None: Met à jour les poids et biais in-place
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for couche in range(L, 0, -1):
        A_prev = cache['A' + str(couche - 1)]

        # Calcul des gradients avec régularisation L2
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        dW += (lambtha / m) * weights['W' + str(couche)]
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if couche > 1:
            W = weights['W' + str(couche)]
            # Calcul de dZ pour la couche précédente avec activation tanh
            dZ = np.matmul(W.T, dZ) * \
                (1 - np.square(cache['A' + str(couche - 1)]))

        # Mise à jour des poids et biais
        weights['W' + str(couche)] -= alpha * dW
        weights['b' + str(couche)] -= alpha * db
