#!/usr/bin/env python3
"""Étape de maximisation pour l'algorithme EM des GMM"""

import numpy as np


def maximization(X, g):
    """
    Réalise l'étape de maximisation pour mettre à jour les paramètres d'un GMM.
    X : matrice de données, de forme (n, d) avec
    n points et d caractéristiques.
    g : matrice des responsabilités, de forme (k, n) où k est
    le nombre de clusters.
    Retourn:
      pi : les poids de chaque cluster, de forme (k,)
      m  : les moyennes de chaque cluster, de forme (k, d)
      S  : les matrices de covariance de chaque cluster, de forme (k, d, d)
    """
    # Vérifie que X est un tableau numpy 2D
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    # Vérifie que g est un tableau numpy 2D
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    # Vérifie que g contient au moins une ligne (un cluster)
    if g.shape[0] < 1:
        return None, None, None
    # Vérifie que le nombre de colonnes de g (nombre de points) correspond au
    # nombre de lignes de X
    if g.shape[1] != X.shape[0]:
        return None, None, None

    n, d = X.shape      # n : nombre d'échantillons, d : dimensions des données
    k = g.shape[0]      # k : nombre de clusters

    # Initialisation des paramètres
    pi = np.zeros(k)         # Poids de chaque cluster
    m = np.zeros((k, d))       # Moyennes de chaque cluster
    S = np.zeros((k, d, d))    # Covariances de chaque cluster

    # Mise à jour des paramètres pour chaque cluster
    for i in range(k):
        # Mise à jour de la proportion du cluster i
        pi[i] = 1 / n * np.sum(g[i], axis=0)

        # Calcul des moyennes pour tous les clusters
        # Cette ligne est exécutée à chaque itération mais calcule les moyennes
        # pour tous les clusters.
        m = np.sum(g[:, :, np.newaxis] * X, axis=1) / \
            np.sum(g, axis=1)[:, np.newaxis]

        # Mise à jour de la matrice de covariance pour le cluster i
        S[i] = (g[i] * (X - m[i]).T @ (X - m[i])) / np.sum(g[i], axis=0)

    # Retourne les nouveaux paramètres estimés du modèle
    return pi, m, S
