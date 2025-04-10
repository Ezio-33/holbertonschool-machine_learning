#!/usr/bin/env python3
"""
Module pour l'algorithme Backward des HMM
Contient la fonction backward
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Calcule les probabilités backward et la vraisemblance des observations

    Args:
        Observation (np.ndarray): Indices des observations (T,)
        Emission (np.ndarray): Matrice d'émission (N, M)
        Transition (np.ndarray): Matrice de transition (N, N)
        Initial (np.ndarray): Probabilités initiales (N, 1)

    Returns:
        tuple: (P, B) où P est la vraisemblance et B la matrice backward
    """
    # Vérification des entrées
    if not all(isinstance(x, np.ndarray)
               for x in [Observation, Emission, Transition, Initial]):
        return None, None
    if Emission.ndim != 2 or Transition.ndim != 2 or Initial.ndim != 2:
        return None, None

    N, M = Emission.shape
    if Transition.shape != (N, N) or Initial.shape != (N, 1):
        return None, None

    T = Observation.size
    if T == 0 or np.any(Observation < 0) or np.any(Observation >= M):
        return None, None

    # Initialisation de la matrice backward
    B = np.zeros((N, T))
    B[:, T - 1] = 1.0  # Cas de base

    # Calcul récursif
    for t in reversed(range(T - 1)):
        obs_next = Observation[t + 1]
        B[:, t] = Transition @ (Emission[:, obs_next] * B[:, t + 1])

    # Calcul de la vraisemblance
    P = np.dot(Initial.ravel(), Emission[:, Observation[0]] * B[:, 0])

    return P, B
