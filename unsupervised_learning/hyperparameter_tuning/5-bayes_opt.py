#!/usr/bin/env python3
"""
Optimisation bayésienne complète avec arrêt sur doublon
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess

class BayesianOptimization:
    """Classe d'optimisation bayésienne avec processus gaussien"""
    
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initialisation des paramètres"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calcule le prochain point optimal avec Expected Improvement"""
        mu, sigma = self.gp.predict(self.X_s)
        sigma = np.maximum(sigma, 1e-10)  # Évite les divisions par zéro

        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0] = 0
        
        X_next = self.X_s[np.argmax(ei)]
        return X_next.reshape(1,), ei

    def optimize(self, iterations=100):
        """Exécute l'optimisation complète avec gestion des doublons"""
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            X_next = X_next.reshape(1, 1)  # Format (1,1) pour comparaison
            
            # Vérification stricte des doublons avec tolérance numérique
            if np.any(np.all(np.isclose(self.gp.X, X_next, atol=1e-8), axis=1)):
                break
                
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        # Sélection du point optimal global selon le mode
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)
            
        return self.gp.X[idx].reshape(1,), self.gp.Y[idx].reshape(1,)
