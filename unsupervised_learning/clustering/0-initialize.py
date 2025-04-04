#!/usr/bin/env python3
"""Initialisation des centroïdes avec distribution uniforme"""

import numpy as np

def initialize(X, k):
    """Initialise les centroïdes selon une distribution uniforme multivariée
    
    Args:
        X : ndarray de forme (n, d)
        k : nombre de clusters
        
    Returns:
        ndarray de forme (k, d)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None

    n, d = X.shape
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    
    return np.random.uniform(low=mins, high=maxs, size=(k, d))
