#!/usr/bin/env python3
"""
Module pour l'algorithme de Viterbi des HMM
Contient la fonction viterbi
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calcule la séquence d'états cachés la plus probable

    Args:
        Observation (np.ndarray): Indices des observations (T,)
        Emission (np.ndarray): Matrice d'émission (N, M)
        Transition (np.ndarray): Matrice de transition (N, N)
        Initial (np.ndarray): Probabilités initiales (N, 1)

    Returns:
        tuple: (chemin, probabilité) ou (None, None) en cas d'erreur
    """
    # Vérifications des entrées
    if not all([isinstance(x, np.ndarray)
               for x in [Observation, Emission, Transition, Initial]]):
        return None, None
    if Emission.ndim != 2 or Transition.ndim != 2 or Initial.ndim != 2:
        return None, None

    N = Transition.shape[0]  # Nombre d'états cachés
    T = Observation.shape[0]  # Nombre d'observations

    # Contrôle des dimensions
    if Transition.shape != (
            N, N) or Emission.shape[0] != N or Initial.shape[0] != N:
        return None, None
    if Initial.shape[1] != 1:
        return None, None

    # Initialisation des matrices delta et psi
    delta = np.zeros((N, T))
    psi = np.zeros((N, T - 1), dtype=int)

    try:
        delta[:, 0] = Initial.flatten() * Emission[:, Observation[0]]
    except IndexError:
        return None, None  # Observation invalide

    # Remplissage des matrices
    for t in range(1, T):
        for j in range(N):
            trans_probs = delta[:, t - 1] * Transition[:, j]
            psi[j, t - 1] = np.argmax(trans_probs)
            delta[j, t] = trans_probs.max() * Emission[j, Observation[t]]

    # Recherche du chemin optimal
    path = [np.argmax(delta[:, T - 1])]
    for t in reversed(range(T - 1)):
        path.append(psi[path[-1], t])

    return path[::-1], np.max(delta[:, T - 1])
