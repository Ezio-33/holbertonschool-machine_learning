#!/usr/bin/env python3
"""
Module pour l'algorithme Forward des HMM
Contient la fonction forward
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Calcule les probabilités forward et la vraisemblance d'une
    séquence d'observations

    Args:
        Observation (np.ndarray): Indices des observations (T,)
        Emission (np.ndarray): Matrice d'émission (N, M)
        Transition (np.ndarray): Matrice de transition (N, N)
        Initial (np.ndarray): Probabilités initiales (N, 1)

    Returns:
        tuple: (P, F) où P est la vraisemblance et F la matrice forward
    """
    # Vérifications des entrées
    if not all([isinstance(x, np.ndarray)
               for x in [Observation, Emission, Transition, Initial]]):
        return None, None

    T = Observation.shape[0]  # Nombre d'observations
    N = Transition.shape[0]   # Nombre d'états cachés

    # Initialisation de la matrice Forward
    F = np.zeros((N, T))
    F[:, 0] = Initial.flatten() * Emission[:, Observation[0]]

    # Calcul récursif
    for t in range(1, T):
        for j in range(N):
            F[j, t] = Emission[j, Observation[t]] * \
                np.sum(Transition[:, j] * F[:, t - 1])

    # Calcul de la vraisemblance totale
    P = np.sum(F[:, -1])

    return P, F
