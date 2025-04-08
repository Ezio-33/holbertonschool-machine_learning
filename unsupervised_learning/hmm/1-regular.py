#!/usr/bin/env python3
"""
Module pour les chaînes de Markov régulières
Contient la fonction regular
"""
import numpy as np


def regular(P):
    """
    Détermine les probabilités stationnaires d'une chaîne de Markov régulière

    Args:
        P (np.ndarray): Matrice de transition (n x n)

    Returns:
        np.ndarray: Vecteur stationnaire (1 x n) ou None si non régulière
    """
    # Vérification de la forme de la matrice
    if not isinstance(
            P,
            np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
        return None

    n = P.shape[0]  # Nombre d'états

    # Calcul de P^300 pour vérifier la régularité
    try:
        P_puissance = np.linalg.matrix_power(P, 300)
    except BaseException:
        return None

    # Vérification des zéros résiduels
    if np.any(P_puissance <= 0):
        return None

    # Extraction de la première ligne (toutes identiques pour une chaîne
    # régulière)
    return P_puissance[0].reshape(1, -1)
