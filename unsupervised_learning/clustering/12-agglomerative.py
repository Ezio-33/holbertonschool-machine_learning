#!/usr/bin/env python3
"""Clustering hiérarchique avec méthode de Ward et dendrogramme"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Effectue un clustering hiérarchique avec seuillage de distance

    Args:
        X: ndarray (n, d) - Données à clusteriser
        dist: float - Distance cophénétique maximale

    Returns:
        ndarray: (n,) - Indices des clusters
    """
    # Calcul de la matrice de linkage
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')

    # Création du dendrogramme avec coloration
    dn = scipy.cluster.hierarchy.dendrogram(
        Z,
        color_threshold=dist,
        above_threshold_color='grey',
        no_labels=True
    )

    # Découpage des clusters selon la distance
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    plt.title('Dendrogramme Ward')
    plt.xlabel('Points de données')
    plt.ylabel('Distance')
    plt.show()

    return clss
