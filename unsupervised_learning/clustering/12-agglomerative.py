#!/usr/bin/env python3
"""12-agglomerative"""

import scipy.cluster.hierarchy as sch  # Import hierarchie de clusters
import matplotlib.pyplot as plt  # Import matplotlib pour affichage graphique

def agglomerative(X, dist):
    # Calcule la matrice de liaison avec la méthode Ward
    linkage_matrix = sch.linkage(X, method="ward")
    # Trace le dendrogramme avec le seuil de couleur donné
    sch.dendrogram(linkage_matrix, color_threshold=dist)
    # Affiche le dendrogramme
    plt.show()
    # Retourne les clusters selon la distance seuil donnée
    return sch.fcluster(linkage_matrix, t=dist, criterion="distance")
