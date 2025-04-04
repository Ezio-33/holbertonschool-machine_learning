#!/usr/bin/env python3
"""Détermine le nombre optimal de clusters par analyse de variance"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance

def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Analyse les clusters de kmin à kmax et calcule les différences de variance
    
    Args:
        X: ndarray de forme (n, d) - données
        kmin: nombre minimal de clusters à tester
        kmax: nombre maximal (auto-détecté si None)
        
    Returns:
        Tuple: (résultats K-means, différences de variance)
    """
    # Validation des entrées
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    n, d = X.shape
    
    # Configuration des bornes
    if kmax is None:
        kmax = n - 1
    if kmin < 1 or kmax > n - 1 or kmin >= kmax:
        return None, None
    
    results = []
    variances = []
    
    # Calcul pour chaque k
    for k in range(kmin, kmax + 1):
        C, _ = kmeans(X, k, iterations)
        if C is None:
            return None, None
        current_var = variance(X, C)
        if current_var is None:
            return None, None
            
        results.append((C, _))
        variances.append(current_var)
    
    # Calcul des différences
    base_var = variances[0]
    d_vars = [base_var - var for var in variances]
    
    return results, np.array(d_vars)
