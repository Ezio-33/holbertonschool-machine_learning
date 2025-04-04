#!/usr/bin/env python3
"""Détermine le meilleur nombre de clusters avec le BIC"""

import numpy as np
# Importation de la fonction EM depuis le module '8-EM'
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Calcule le BIC pour différents nombres de clusters et retourne le meilleur.

    Paramètres:
    - X: np.ndarray, les données.
    - kmin: int, nombre minimal de clusters.
    - kmax: int, nombre maximal de clusters.
    - iterations: int, nombre maximum d'itérations pour EM.
    - tol: float, tolérance pour la convergence.
    - verbose: bool, affichage d'informations supplémentaires.

    Retourne:
    - best_k: int, meilleur nombre de clusters choisi.
    - best_result: tuple, contient pi, m, S du meilleur EM.
    - l: float, log-vraisemblance finale.
    - b: np.ndarray, valeurs de BIC pour chaque k.
    """
    # Récupération des dimensions des données
    n, d = X.shape
    # Initialisation d'un tableau pour log-vraisemblance
    l_arr = np.zeros(kmax - kmin + 1)
    # Initialisation d'un tableau pour le BIC
    b = np.zeros(kmax - kmin + 1)
    # Boucle sur les différentes valeurs de k
    for k in range(kmin, kmax + 1):
        print("K", k)
        # Calcul de EM pour k clusters
        pi, m, S, g, l = expectation_maximization(
            X, k, iterations, tol, verbose=False)
        # Calcul du BIC pour k clusters
        b[k - kmin] = k * np.log(n) - (2 * l)
        # Sauvegarde de la log-vraisemblance dans le tableau
        l_arr[k - kmin] = l
    # Sélection du meilleur k selon le maximum du BIC
    best_k = np.argmax(b) + kmin
    # Stockage de l'indice de la meilleure log-vraisemblance
    best_l = np.argmax(l_arr)
    print("best k", best_k)
    # Sélection des résultats correspondant au meilleur EM
    best_result = pi, m, S
    print("best result", best_result)
    # Renvoi du meilleur k et des résultats associés
    return best_k, best_result, l, b
