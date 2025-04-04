#!/usr/bin/env python3
"""Implémentation de Gaussian Mixture Model avec Scikit-Learn"""

import numpy as np
import sklearn.mixture


def gmm(X, k):
    """Entraîne un modèle GMM et retourne ses paramètres

    Args:
        X (numpy.ndarray): Données de forme (n, d)
        k (int): Nombre de clusters

    Returns:
        tuple: (pi, m, S, clss, bic)
            - pi: Poids des clusters (k,)
            - m: Moyennes des clusters (k, d)
            - S: Covariances des clusters (k, d, d)
            - clss: Affectations des points (n,)
            - bic: Valeur BIC du modèle
    """
    # Création et entraînement du modèle
    model = sklearn.mixture.GaussianMixture(n_components=k)
    model.fit(X)

    # Extraction des paramètres
    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
