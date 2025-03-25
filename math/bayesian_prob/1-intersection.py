#!/usr/bin/env python3
"""Module de calcul d'intersection bayésienne"""
import numpy as np

def intersection(x, n, P, Pr):
    """
    Calcule l'intersection entre données
	observées et probabilités hypothétiques
    
    Args:
        x: Nombre de patients avec effets secondaires
        n: Nombre total de patients
        P: Tableau des probabilités hypothétiques
        Pr: Tableau des croyances a priori
        
    Returns:
        Tableau numpy des intersections
    """
    # Vérification dans l'ordre exact des tests
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    
    if x > n:
        raise ValueError("x cannot be greater than n")
    
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    
    # Validation séparée de P et Pr
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
        
    if np.any((Pr < 0) | (Pr > 1)):  # Nouvelle vérification spécifique à Pr
        raise ValueError("All values in Pr must be in the range [0, 1]")
    
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calcul optimisé avec gestion des cas limites
    results = np.zeros_like(P)

    # Cas où P est entre 0 et 1 exclus
    mask = (P > 0) & (P < 1)
    valid_P = P[mask]

    if valid_P.size > 0:
        # Calcul du coefficient binomial avec log pour éviter les overflows
        log_coef = 0
        for i in range(1, x + 1):
            log_coef += np.log(n - x + i) - np.log(i)

        # Calcul de la vraisemblance
        results[mask] = np.exp(
            log_coef
            + x * np.log(valid_P)
            + (n - x) * np.log(1 - valid_P)
        )

    # Cas spéciaux P=0 et P=1
    results[P == 0] = 0.0 if x > 0 else 1.0
    results[P == 1] = 0.0 if x < n else 1.0

    # Calcul final de l'intersection
    return results * Pr
