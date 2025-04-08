#!/usr/bin/env python3
"""
Module pour les calculs de chaînes de Markov
Contient la fonction markov_chain
"""
import numpy as np  # Import de numpy pour les calculs matriciels


def markov_chain(P, s, t=1):
    """
    Calcule la probabilité d'être dans chaque état après t étapes

    Args:
        P (numpy.ndarray): Matrice de transition (n x n)
        s (numpy.ndarray): Vecteur d'état initial (1 x n)
        t (int): Nombre d'étapes (défaut: 1)

    Returns:
        numpy.ndarray: Vecteur de probabilités final (1 x n) ou None en cas d'erreur
    """
    # Vérification des entrées
    if not isinstance(
            P,
            np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
        return None  # P n'est pas une matrice carrée
    if not isinstance(
            s,
            np.ndarray) or s.ndim != 2 or s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None  # s n'a pas la bonne forme
    if not isinstance(t, int) or t < 1:
        return None  # t n'est pas un entier positif

    # Calcul de la matrice P^t
    P_puissance = np.linalg.matrix_power(P, t)

    # Calcul du vecteur résultant s * P^t
    return s @ P_puissance  # @ représente la multiplication matricielle
