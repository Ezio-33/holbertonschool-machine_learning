#!/usr/bin/env python3
"""
Module qui calcule l'intégrale d'un polynôme
"""


def poly_integral(poly, C=0):
    """
    Args:
        poly: Liste des coefficients du polynôme
        C: Constante d'intégration (par défaut 0)

    Return:
        Liste représentant l'intégrale du polynôme
        None: Si poly n'est pas valide ou si C n'est pas un entier
    """
    if not isinstance(poly, list) or not isinstance(C, int):
        return None

    if len(poly) == 0:
        return None

    # Calcul de l'intégrale
    integral = [C]  # Commence avec la constante d'intégration
    for i in range(len(poly)):
        coef = poly[i] / (i + 1)
        # Si le coefficient est un entier, le convertir
        if coef.is_integer():
            coef = int(coef)
        if coef != 0:
            integral.append(coef)

    return integral
