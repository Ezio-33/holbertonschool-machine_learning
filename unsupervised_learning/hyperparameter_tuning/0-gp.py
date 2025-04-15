#!/usr/bin/env python3
"""
Module implémentant un processus gaussien 1D sans bruit avec noyau RBF
"""

import numpy as np


class GaussianProcess:
    """
    Classe représentant un processus gaussien avec noyau RBF

    Attributs :
        X (np.ndarray) : Points d'entrée observés (forme (t, 1))
        Y (np.ndarray) : Sorties correspondantes (forme (t, 1))
        l (float) : Paramètre de longueur du noyau
        sigma_f (float) : Amplitude du noyau
        K (np.ndarray) : Matrice de covariance actuelle
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initialise le processus gaussien

        Args:
            X_init (np.ndarray): Points initiaux (forme (t, 1))
            Y_init (np.ndarray): Valeurs observées (forme (t, 1))
            l (float): Paramètre de longueur (défaut 1)
            sigma_f (float): Amplitude (défaut 1)
        """
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
