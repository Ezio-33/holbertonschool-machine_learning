#!/usr/bin/env python3
"""
Propagation avant avec Dropout
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Effectue la propagation avant avec Dropout.

    Args:
        X (np.ndarray): Données d'entrée (taille nx × m)
        weights (dict): Dictionnaire des poids et biais
        L (int): Nombre total de couches
        keep_prob (float): Probabilité de conserver un neurone

    Returns:
        dict: Cache contenant les activations et masques Dropout
    """
    cache = {'A0': X.copy()}  # Stockage de l'entrée
    A_prev = X  # Initialisation avec l'entrée

    # Parcours des couches cachées (1 à L-1)
    for layer in range(1, L):
        W = weights[f'W{layer}']
        b = weights[f'b{layer}']
        Z = np.dot(W, A_prev) + b  # Calcul linéaire
        A = np.tanh(Z)  # Activation tanh

        # Génération du masque Dropout et application
        D = np.random.binomial(1, keep_prob, size=A.shape)
        A = (A * D) / keep_prob  # Ajustement d'échelle

        # Mise à jour du cache
        cache[f'A{layer}'] = A
        cache[f'D{layer}'] = D
        A_prev = A  # Préparation pour la couche suivante

    # Couche de sortie (pas de Dropout)
    W = weights[f'W{L}']
    b = weights[f'b{L}']
    Z = np.dot(W, A_prev) + b
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)  # Softmax
    cache[f'A{L}'] = A

    return cache
