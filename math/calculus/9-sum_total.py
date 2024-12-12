#!/usr/bin/env python3
"""    Calcule la somme des carrés des nombres de 1 à n
"""


def summation_i_squared(n):
    """
    Args:
        n: nombre entier positif
    Return:
        somme des carrés ou None si n invalide
    """
    if not isinstance(n, int) or n < 1:
        return None
    # Formule : sum(i²) = (n * (n + 1) * (2n + 1)) / 6
    return (n * (n + 1) * (2 * n + 1)) // 6
