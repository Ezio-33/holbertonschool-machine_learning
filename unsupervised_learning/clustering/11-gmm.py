#!/usr/bin/env python3
"""11-gmm.py: Calcule un modèle GMM sur un dataset X avec k composantes."""

import sklearn.mixture as mix


def gmm(X, k):
    """
    Calcule le modèle Gaussien à mélanges (GMM) sur X.

    Paramètres:
      X (array-like): Données pour entraînement.
      k (int): Nombre de composantes du modèle.

    Retourne:
      tuple: Poids, moyennes, covariances, prédictions, bic.
    """
    # Crée et entraîne le modèle GMM avec k composants
    model = mix.GaussianMixture(n_components=k).fit(X)

    # Retourne les paramètres du modèle entraîné:
    # - Poids de chaque composante
    # - Moyennes de chaque composante
    # - Covariances de chaque composante
    # - Prédictions pour X
    # - Critère Bayesian Information (BIC)
    return (model.weights_, model.means_,
            model.covariances_, model.predict(X),
            model.bic(X))
