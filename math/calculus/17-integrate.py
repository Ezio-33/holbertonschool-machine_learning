#!/usr/bin/env python3
"""
Module qui calcule l'intégrale d'un polynôme
"""


def poly_integral(poly, C=0):
    """
    Calcule l'intégrale d'un polynôme

    Args:
        poly: Liste des coefficients du polynôme
        C: Constante d'intégration (par défaut 0)

    Return:
        Liste représentant l'intégrale du polynôme
        None: Si poly n'est pas valide ou si C n'est pas un entier
    """
    if not isinstance(poly, list) or not isinstance(C, int):
        return None

    if not poly:
        return None

    # Cas spécial pour polynôme nul
    if len(poly) == 1 and poly[0] == 0:
        return [0]

    # Calcul de l'intégrale
    result = [C]
    for power, coef in enumerate(poly):
        # Division par la puissance + 1
        term = coef / (power + 1)
        # Conversion en entier si possible
        result.append(int(term) if term.is_integer() else term)

    # Suppression des zéros non significatifs à la fin que si nécessaire.
    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result
