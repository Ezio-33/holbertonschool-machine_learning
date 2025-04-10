#!/usr/bin/env python3
"""
Module pour l'algorithme de Baum-Welch des HMM
Contient la fonction baum_welch
"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Estime les paramètres d'un HMM en utilisant l'algorithme de Baum-Welch

    Args:
        Observations (np.ndarray): Indices des observations (T,)
        Transition (np.ndarray): Matrice de transition initiale (N, N)
        Emission (np.ndarray): Matrice d'émission initiale (N, M)
        Initial (np.ndarray): Probabilités initiales (N, 1)
        iterations (int): Nombre d'itérations EM

    Returns:
        tuple: (Transition mise à jour, Emission mise à jour)
    """
    # Vérification des entrées
    if not all(isinstance(arr, np.ndarray)
               for arr in [Observations, Transition, Emission, Initial]):
        return None, None
    N, M = Emission.shape
    T = Observations.size

    # Copie pour éviter la modification des entrées
    Transition = Transition.copy().astype(float)
    Emission = Emission.copy().astype(float)
    Initial = Initial.copy().astype(float)

    for _ in range(iterations):
        # Etape E : Calcul des probabilités forward et backward
        # Forward
        alpha = np.zeros((N, T))
        alpha[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
        for t in range(1, T):
            for j in range(N):
                alpha[j, t] = Emission[j, Observations[t]] * \
                    np.sum(Transition[:, j] * alpha[:, t - 1])

        # Backward
        beta = np.zeros((N, T))
        beta[:, -1] = 1.0
        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[i, t] = np.sum(
                    Transition[i, :] *
                    Emission[:, Observations[t + 1]] *
                    beta[:, t + 1])

        # Calcul de la probabilité totale
        P = np.sum(alpha[:, -1])
        if P == 0:
            return None, None

        # Calcul de gamma (probabilité d'être dans l'état i au temps t)
        gamma = (alpha * beta) / P

        # Calcul de xi (probabilité de transition i->j au temps t)
        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            obs_next = Observations[t + 1]
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = alpha[i, t] * Transition[i, j] * \
                        Emission[j, obs_next] * beta[j, t + 1]
            xi[t] /= P  # Normalisation

        # Etape M : Mise à jour des paramètres
        # Mise à jour de la matrice de transition
        Transition_new = np.zeros((N, N))
        for i in range(N):
            denom = np.sum(gamma[i, :-1])  # Somme sur t=0 à T-2
            if denom == 0:
                continue
            for j in range(N):
                Transition_new[i, j] = np.sum(xi[:, i, j]) / denom

        # Mise à jour de la matrice d'émission
        Emission_new = np.zeros((N, M))
        for j in range(N):
            denom = np.sum(gamma[j])
            if denom == 0:
                continue
            for k in range(M):
                Emission_new[j, k] = np.sum(
                    gamma[j, Observations == k]) / denom

        # Normalisation des lignes
        Transition_new /= Transition_new.sum(axis=1, keepdims=True)
        Emission_new /= Emission_new.sum(axis=1, keepdims=True)

        Transition, Emission = Transition_new, Emission_new

    return Transition, Emission
