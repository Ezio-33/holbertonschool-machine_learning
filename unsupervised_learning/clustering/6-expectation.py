#!/usr/bin/env python3
"""Étape d'expectation pour l'algorithme EM des GMM"""

import numpy as np
pdf = __import__('5-pdf').pdf

def expectation(X, pi, m, S):
    """Calcule les probabilités postérieures et la log-vraisemblance
    
    Args:
        X : ndarray (n, d) - Données
        pi : ndarray (k,) - Priors des clusters
        m : ndarray (k, d) - Moyennes des clusters
        S : ndarray (k, d, d) - Matrices de covariance
        
    Returns:
        tuple: (g, l) où g est (k, n) des probabilités postérieures,
               l est la log-vraisemblance totale
    """
    # Validation des entrées
    if not all([isinstance(arr, np.ndarray) for arr in [X, pi, m, S]]):
        return None, None
    if X.ndim != 2 or pi.ndim != 1 or m.ndim != 2 or S.ndim != 3:
        return None, None
    
    k = pi.shape[0]
    n, d = X.shape
    
    # Vérification cohérence des dimensions
    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None
    
    g = np.zeros((k, n))
    
    # Calcul des PDF pondérées pour chaque cluster
    for i in range(k):
        pdf_vals = pdf(X, m[i], S[i])
        if pdf_vals is None:
            return None, None
        g[i] = pi[i] * pdf_vals
    
    # Somme des probabilités par point (dénominateur)
    denom = np.sum(g, axis=0)
    denom = np.maximum(denom, 1e-300)  # Éviter division par zéro
    
    # Normalisation pour obtenir les probabilités postérieures
    g /= denom
    
    # Calcul de la log-vraisemblance totale
    log_likelihood = np.sum(np.log(denom))
    
    return g, log_likelihood
