#!/usr/bin/env python3
"""Implémentation de K-means avec Scikit-Learn"""

import numpy as np
from sklearn.cluster import KMeans


def kmeans(X, k):
    """Effectue un clustering K-means avec Scikit-Learn

    Args:
        X (numpy.ndarray): Données de forme (n, d)
        k (int): Nombre de clusters

    Returns:
        tuple: (C, clss)
            - C : centroïdes des clusters (k, d)
            - clss : indices des clusters pour chaque point (n,)
    """
    # Création du modèle K-means avec configuration de base
    kmeans_model = KMeans(n_clusters=k, n_init=10)

    # Entraînement du modèle sur les données
    kmeans_model.fit(X)

    # Récupération des résultats
    return kmeans_model.cluster_centers_, kmeans_model.labels_
