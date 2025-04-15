#!/usr/bin/env python3
"""
Module implémentant la prédiction avec un processus gaussien 1D
"""

import numpy as np


class GaussianProcess:
    """
    Classe pour les prédictions d'un processus gaussien avec noyau RBF

    Attributs :
        X (np.ndarray): Points d'entrée observés (forme (t, 1))
        Y (np.ndarray): Sorties correspondantes (forme (t, 1))
        l (float): Paramètre de longueur du noyau
        sigma_f (float): Amplitude du noyau
        K (np.ndarray): Matrice de covariance actuelle
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialisation avec les points d'entraînement"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calcule le noyau RBF entre deux ensembles de points

        Args:
            X1 (np.ndarray): Première matrice de points (m, 1)
            X2 (np.ndarray): Deuxième matrice de points (n, 1)

        Returns:
            np.ndarray: Matrice de covariance (m, n)
        """
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
            np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Prédit la moyenne et la variance pour de nouveaux points

        Args:
            X_s (np.ndarray): Points à prédire (forme (s, 1))

        Returns:
            tuple: (moyennes (s,), variances (s,))
        """
        # Matrice de covariance entre points connus et nouveaux points
        K_s = self.kernel(self.X, X_s)

        # Inverse de la matrice de covariance existante
        K_inv = np.linalg.inv(self.K)

        # Calcul de la moyenne
        mu = K_s.T.dot(K_inv).dot(self.Y).squeeze()

        # Calcul de la variance
        cov_s = self.kernel(X_s, X_s) - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(cov_s)

        return mu, sigma
