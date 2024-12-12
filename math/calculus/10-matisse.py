#!/usr/bin/env python3
"""
Module qui calcule la dérivée d'un polynôme
"""


def poly_derivative(poly):
    """
    Calcule la dérivée d'un polynôme

    Args:
        Liste des coefficients du polynôme
            index 0 = terme constant
            index 1 = coefficient de x
            index 2 = coefficient de x²
            etc.

    Return:
        Liste des coefficients de la dérivée
        None: Si poly n'est pas valide
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # Si le polynôme est constant
    if len(poly) == 1:
        return [0]

    # Calcul de la dérivée
    derivative = []
    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)

    # Si tous les coefficients sont 0
    if all(coef == 0 for coef in derivative):
        return [0]

    return derivative
