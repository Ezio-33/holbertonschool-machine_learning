#!/usr/bin/env python3
"""
Module d'optimisation bayésienne basée sur un processus gaussien
"""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Classe pour l'optimisation bayésienne de fonctions boîte noire

    Attributs :
        f (function): Fonction à optimiser
        gp (GaussianProcess): Processus gaussien
        X_s (np.ndarray): Points d'acquisition (forme (ac_samples, 1))
        xsi (float): Facteur d'exploration-exploitation
        minimize (bool): Mode minimisation (True) ou maximisation (False)
    """

    def __init__(
            self,
            f,
            X_init,
            Y_init,
            bounds,
            ac_samples,
            l=1,
            sigma_f=1,
            xsi=0.01,
            minimize=True):
        """
        Initialise l'optimisation bayésienne

        Args:
            f (function): Fonction à optimiser
            X_init (np.ndarray): Points initiaux (forme (t, 1))
            Y_init (np.ndarray): Valeurs initiales (forme (t, 1))
            bounds (tuple): Bornes (min, max) de l'espace de recherche
            ac_samples (int): Nombre de points d'acquisition
            l (float): Longueur caractéristique du noyau
            sigma_f (float): Amplitude du processus gaussien
            xsi (float): Paramètre d'exploration
            minimize (bool): Mode optimisation (True = minimisation)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)

        # Génération des points d'acquisition
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize
