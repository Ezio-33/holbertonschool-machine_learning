#!/usr/bin/env python3
"""
Module d'optimisation bayésienne complète avec sélection du point optimal
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Optimisation bayésienne complète avec arrêt prématuré sur doublons

    Attributs:
        gp (GaussianProcess): Processus gaussien
        X_s (np.ndarray): Points d'exploration
        xsi (float): Paramètre d'exploration
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
        """Initialisation identique aux tâches précédentes"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
                Fonction d'acquisition Expected Improvement"""
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        with np.errstate(divide='ignore'):
            Z = imp / sigma
            Z[sigma == 0] = 0

        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0] = 0

        X_next = self.X_s[np.argmax(ei)]
        return X_next.reshape(1,), ei.flatten()

    def optimize(self, iterations=100):
        """
        Exécute l'optimisation bayésienne complète

        Args:
            iterations (int): Nombre maximal d'itérations

        Returns:
            tuple: (X_opt, Y_opt) point optimal et sa valeur
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()  # X_next shape (1,)
            X_next_scalar = X_next[0]  # Conversion en scalaire

            # Vérification des doublons avec tolérance numérique
            if np.any(
                np.isclose(
                    self.gp.X.flatten(),
                    X_next_scalar,
                    atol=1e-6)):
                break

            # Évaluation de la fonction black-box
            X_next_2d = X_next.reshape(-1, 1)  # Format 2D pour compatibilité
            Y_next = self.f(X_next_2d)

            # Mise à jour du processus gaussien
            self.gp.update(X_next_2d, Y_next.reshape(-1, 1))

        # Sélection du point optimal selon le mode
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx].reshape(1,)  # Format de sortie
        Y_opt = self.gp.Y[idx].reshape(1,)

        return X_opt, Y_opt
