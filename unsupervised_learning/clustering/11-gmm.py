#!/usr/bin/env python3
"""Implémentation de Gaussian Mixture Model avec Scikit-Learn"""

import sklearn.cluster  # Importation du module clustering de scikit-learn


def kmeans(X, k):
    """
    Applique KMeans sur les données X pour k clusters.

    Paramètres:
        X -- données d'entrée (matrice ou array)
        k -- nombre de clusters à trouver

    Retourne:
        C     -- centres des clusters
        clss  -- étiquettes attribuées à chaque point
    """
    # Applique l'algorithme KMeans avec k clusters sur X
    kmean = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    # Centre de chaque cluster calculé par l'algorithme
    C = kmean.cluster_centers_
    # Label attribué à chaque échantillon en fonction du cluster
    clss = kmean.labels_

    # Retourne les centres de clusters et les labels correspondants
    return C, clss
