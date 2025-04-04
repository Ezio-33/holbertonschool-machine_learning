#!/usr/bin/env python3
"""Implémentation de K-means optimisée avec gestion
précise des clusters vides"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Applique l'algorithme de K-means sur l'ensemble de points X

    Paramètres:
      X : np.ndarray de forme (n, d), n points dans un espace de dimension d
      k : int, nombre de clusters
      iterations : int, nombre maximum d'itérations

    Retour:
      centroid : np.ndarray de forme (k, d) contenant les centroïdes finaux
      clss : np.ndarray de dimension (n,) contenant l'indice
      du cluster pour chaque point
    """

    # Vérifier que X est un tableau 2D
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    # Vérifier que k est un entier positif
    if not isinstance(k, int) or k <= 0:
        return None, None

    # Vérifier que iterations est un entier supérieur à 0
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    n, d = X.shape  # n : nombre de points, d : dimension de chaque point

    # Initialiser k centroïdes de manière aléatoire dans l'intervalle des
    # valeurs de X
    centroid = np.random.uniform(low=np.min(X, axis=0),
                                 high=np.max(X, axis=0), size=(k, d))

    # Boucle principale de l'algorithme de K-means
    for i in range(iterations):
        # Calculer la distance euclidienne entre chaque point et chaque
        # centroïde
        distances = np.linalg.norm(X[:, np.newaxis] - centroid, axis=2)

        # Assigner chaque point au cluster avec le centroïde le plus proche
        clss = np.argmin(distances, axis=1)

        # Faire une copie des centroïdes actuels pour vérifier l'arrêt de
        # l'algorithme
        new_centroid = np.copy(centroid)

        # Mettre à jour chaque centroïde en fonction des points assignés
        for j in range(k):
            # Si aucun point n'est assigné au cluster j
            if len(np.where(clss == j)[0]) == 0:
                # Réinitialiser le centroïde j de manière aléatoire
                centroid[j] = np.random.uniform(np.min(X, axis=0),
                                                np.max(X, axis=0), d)
            else:
                # Mettre à jour le centroïde j en calculant la moyenne des
                # points du cluster
                centroid[j] = np.mean(X[np.where(clss == j)], axis=0)

        # Si les centroïdes n'ont pas changé, arrêter l'algorithme
        if np.array_equal(new_centroid, centroid):
            break

    # Recalculer les distances et l'affectation finale des points aux clusters
    distances = np.linalg.norm(X[:, np.newaxis] - centroid, axis=2)
    clss = np.argmin(distances, axis=1)

    return centroid, clss
