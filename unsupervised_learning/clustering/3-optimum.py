#!/usr/bin/env python3
"""Détermine le nombre optimal de clusters par analyse de variance"""

import numpy as np
# Importation de la fonction kmeans à partir du module '1-kmeans'
kmeans = __import__('1-kmeans').kmeans
# Importation de la fonction variance à partir du module '2-variance'
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Analyse de la variance en fonction du nombre de grappes.

    Paramètres:
    X : np.ndarray
        Tableau de données à regrouper (shape: n x d).
    kmin : int
        Nombre minimal de clusters à tester.
    kmax : int ou None
        Nombre maximal de clusters à tester; si None, il sera défini plus tard.
    iterations : int
        Nombre d'itérations pour la fonction kmeans.

    Retourne:
    results : list
        Liste des résultats de kmeans sous forme de tuples
        (C, clss) pour chaque k.
    d_var : list
        Liste des variations de variance entre le premier cluster et chaque k.
    """
    # Vérification que X est un tableau numpy à deux dimensions
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    n, d = X.shape  # n: nombre d'exemples, d: dimension des données

    # Vérification que kmin est un entier >= 1
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    # Définition de kmax si non fourni
    if kmax is None:
        kmax = iterations
    # Vérification que kmax est un entier >= 1
    if not isinstance(kmax, int) or kmax < 1:
        return None, None
    # kmin doit être strictement inférieur à kmax pour que la boucle ait un
    # sens
    if kmin >= kmax:
        return None, None

    results = []  # Liste pour stocker les résultats de la fonction kmeans
    d_var = []    # Liste pour stocker la différence de variance

    # Ajuste kmax pour ne pas dépasser le nombre d'exemples
    if kmax is None or kmax >= n:
        kmax = n

    # Boucle sur chaque nombre de clusters compris entre kmin et kmax inclus
    for k in range(kmin, kmax + 1):
        # Exécute kmeans sur les données X avec k clusters et nombre
        # d'itérations spécifié
        C, clss = kmeans(X, k, iterations)
        # Calcul de la variance totale au sein des clusters
        var = np.sum((X[:] - C[clss]) ** 2)

        # Pour le premier k, initialise la variance de référence (var_min)
        if k == kmin:
            var_min = var
        # Ajoute le résultat (centres et classes) dans la liste results
        results.append((C, clss))
        # Calcule la différence de variance avec la variance de référence et
        # l'ajoute dans d_var
        d_var.append(var_min - var)

    # Retourne les résultats de kmeans et les variations de variance associées
    return results, d_var
