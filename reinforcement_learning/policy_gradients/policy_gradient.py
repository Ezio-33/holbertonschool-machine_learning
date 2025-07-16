#!/usr/bin/env python3
"""
Module de Gradient de Politique pour l'apprentissage par renforcement
"""

import numpy as np


def policy(matrix, weight):
    """
    Calcule la politique avec un poids pour une matrice.

    Args:
        matrix: matrice représentant l'observation actuelle de l'environnement
        weight: matrice de poids aléatoires

    Returns:
        Les probabilités de politique pour chaque action
    """
    # Assure que la matrice est en 2D
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    # Calcule le produit scalaire entre l'état et les poids
    z = np.dot(matrix, weight)

    # Applique softmax pour obtenir les probabilités
    exp_z = np.exp(z)
    policy_probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return policy_probs


def policy_gradient(state, weight):
    """
    Calcule le gradient de politique Monte-Carlo basé sur un état et un poids.

    Args:
        state: matrice représentant l'observation actuelle de l'environnement
        weight: matrice de poids aléatoires

    Returns:
        L'action et le gradient (dans cet ordre)
    """
    # Assure que l'état est en 2D
    if state.ndim == 1:
        state = state.reshape(1, -1)

    # Obtient les probabilités de politique
    policy_probs = policy(state, weight)

    # Échantillonne une action à partir de la distribution de politique
    action = np.random.choice(len(policy_probs[0]), p=policy_probs[0])

    # Calcule le gradient du log de la politique: ∇log(π(a|s))
    # Pour politique softmax: ∇log(π(a|s)) = x(s) * (δ(a) - π(a|s))
    # où δ(a) = 1 si action a est sélectionnée, 0 sinon
    gradient = np.zeros_like(weight)

    # Crée un vecteur one-hot pour l'action sélectionnée
    action_onehot = np.zeros(weight.shape[1])
    action_onehot[action] = 1

    # Calcule gradient: produit externe état et (one-hot - probs)
    gradient = np.outer(state[0], action_onehot - policy_probs[0])

    return action, gradient
