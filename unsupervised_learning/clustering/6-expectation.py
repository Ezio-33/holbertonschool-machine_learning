#!/usr/bin/env python3
"""Étape d'expectation pour l'algorithme EM des GMM"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calcule l'étape d'expectation de l'algorithme EM pour les
    mixtures gaussiennes.

    Paramètres:
    - X (np.ndarray, shape=(n, d)): Ensemble des données avec n
    observations et d dimensions.
    - pi (np.ndarray, shape=(k,)): Probabilités a priori pour
    chaque cluster (grappe).
    - m (np.ndarray, shape=(k, d)): Moyennes des centroïdes pour
    chaque cluster.
    - S (np.ndarray, shape=(k, d, d)): Matrices de covariance
    pour chaque cluster.

    Retourne:
    - g (np.ndarray, shape=(k, n)): Matrice de postériorités pour chaque
    cluster et chaque donnée.
    - likelihood (float): Logarithme de la vraisemblance totale des données.
    """
    # Vérifier que X est bien un tableau 2D
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    n, d = X.shape  # n: nombre d'observations, d: dimension des données

    # Vérifier que pi est bien un vecteur 1D
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None

    k = pi.shape[0]  # k: nombre de clusters

    # Vérifier que les priors sont correctement définis
    if k <= 0 or not np.isclose(np.sum(pi), 1):
        return None, None

    # Vérifier que m est un tableau 2D
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None

    # Vérifier que S est un tableau 3D et que chaque matrice de covariance est
    # carrée
    if not isinstance(
            S, np.ndarray) or len(
            S.shape) != 3 or S.shape[1] != S.shape[2]:
        return None, None

    # Vérifier la cohérence des dimensions des moyennes et des covariances
    # avec les données
    if m.shape[1] != d or S.shape[2] != d:
        return None, None

    # Vérifier que le nombre de clusters est le même pour pi, m et S
    if pi.shape[0] != k or S.shape[0] != k or m.shape[0] != k:
        return None, None

    # Initialisation de la matrice de probabilités pondérées de chaque cluster
    # pour chaque donnée
    g = np.zeros((k, n))
    # Contiendra la somme sur tous les clusters pour normaliser
    sigma_g = np.zeros(n)

    # Calculer le produit de la probabilité a priori et de la densité pdf pour
    # chaque cluster
    for i in range(k):
        # Calcul de la densité de probabilité pour le cluster i
        pdf_0 = pdf(X, m[i], S[i])
        # Pondération par la probabilité a priori du cluster i
        g[i] = pi[i] * pdf_0
        # Somme pour toutes les observations
        sigma_g += g[i]

    # Normalisation pour obtenir la postériorité
    g = g / sigma_g

    # Calculer la vraisemblance totale: somme des logarithmes de la somme des
    # probabilités (pour éviter le underflow)
    likelihood = np.sum(np.log(sigma_g))

    return g, likelihood
