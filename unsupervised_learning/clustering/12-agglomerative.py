#!/usr/bin/env python3
"""Clustering hiérarchique avec méthode de Ward et dendrogramme"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Applique l'algorithme d'agglomération sur les données X.

    Paramètres:
    X: array-like, données à regrouper.
    dist: float, seuil de distance pour le dendrogramme.

    Retourne:
    clss: array, étiquettes des clusters trouvés.
    """
    # Calcul de la matrice de linkage avec la méthode de Ward
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')

    # Création du dendrogramme utilisant le seuil pour les couleurs
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)

    # Affichage du dendrogramme
    plt.show()

    # Attribution des clusters selon la hauteur 'dist'
    clss = scipy.cluster.hierarchy.fcluster(Z, dist, criterion='distance')
    return clss
