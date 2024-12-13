#!/usr/bin/env python3
"""
Module qui calcule l'intégrale d'un polynôme
"""


def poly_integral(poly, C=0):
    """
    Calcule l'intégrale d'un polynôme

    Args:
        poly (list): Liste des coefficients du polynôme
        C (int): Constante d'intégration (par défaut 0)

    Returns:
        list: Liste représentant l'intégrale du polynôme
        None: Si poly n'est pas valide ou si C n'est pas un entier
    """
    if not isinstance(poly, list) or not isinstance(C, int):
        return None

    if len(poly) == 0:
        return None

    integral = [C]
    for i, coef in enumerate(poly):
        if not isinstance(coef, (int, float)):
            return None
        term = coef / (i + 1)
        integral.append(int(term) if term.is_integer() else term)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
