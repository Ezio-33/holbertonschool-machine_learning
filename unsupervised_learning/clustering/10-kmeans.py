#!/usr/bin/env python3
"""Implémentation de K-means avec Scikit-Learn

Cette fonction calcule les clusters d'un ensemble de données X 
en utilisant l'algorithme K-means. 
Elle renvoie les centres des clusters et les labels des données.
"""

import sklearn.cluster  # Importation du module de clustering de Scikit-Learn


def kmeans(X, k):
    """
    Réalise le clustering de type K-means sur l'ensemble de données X.
    
    Paramètres:
        X (array-like): Les données d'entrée à clusteriser.
        k (int): Le nombre de clusters à former.
    
    Retourne:
        tuple: Un tuple contenant:
            - C (array-like): Les centres des clusters.
            - clss (array-like): Les labels indiquant à
            quel cluster chaque donnée appartient.
    """
    # Création d'un modèle KMeans avec k clusters
    model = sklearn.cluster.KMeans(n_clusters=k)
    
    # Entraînement du modèle sur les données X
    model.fit(X)
    
    # Récupération des centres des clusters
    C = model.cluster_centers_
    
    # Attribution des labels à chaque point dans X
    clss = model.labels_
    
    # Retourner les centres des clusters et les labels correspondants
    return C, clss
