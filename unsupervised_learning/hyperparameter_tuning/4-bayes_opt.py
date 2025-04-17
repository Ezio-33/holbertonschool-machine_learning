#!/usr/bin/env python3
"""
Module implémentant la fonction d'acquisition Expected Improvement
"""

import numpy as np
from scipy.stats import norm


class BayesianOptimization:
    """
    Classe pour l'optimisation bayésienne avec fonction d'acquisition

    Attributs:
        gp (GaussianProcess): Processus gaussien initialisé
        X_s (np.ndarray): Points d'exploration
        xsi (float): Facteur exploration/exploitation
        minimize (bool): Mode minimisation
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
        """Initialisation identique à la tâche 3"""
        self.f = f
        self.gp = __import__(
            '2-gp').GaussianProcess(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calcule l'Expected Improvement (EI) pour tous les points d'exploration

        Returns:
            tuple: (X_next, EI)
            X_next (np.ndarray): Point optimal suivant (forme (1,))
            EI (np.ndarray): Valeurs EI pour tous les points
                        (forme (ac_samples,))
        """
        # Étape 1: Prédiction sur les points d'exploration
        mu, sigma = self.gp.predict(self.X_s)

        # Étape 2: Détermination de la meilleure valeur actuelle
        if self.minimize:
            best = np.min(self.gp.Y)
            improvement = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            improvement = mu - best - self.xsi

        # Étape 3: Calcul de Z avec gestion des divisions par zéro
        with np.errstate(divide='ignore'):
            Z = improvement / sigma
            Z[sigma == 0] = 0  # Évite les NaN

        # Étape 4: Calcul de l'Expected Improvement
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0] = 0  # Cas où sigma=0

        # Étape 5: Sélection du point optimal
        index = np.argmax(ei)
        X_next = self.X_s[index]

        return X_next.reshape(1,), ei.flatten()
