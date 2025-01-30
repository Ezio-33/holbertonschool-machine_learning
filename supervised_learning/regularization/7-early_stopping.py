#!/usr/bin/env python3
"""
Module pour l'early stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Détermine si l'entraînement doit être arrêté prématurément.

    Args:
        cost (float): Coût actuel de validation.
        opt_cost (float): Meilleur coût enregistré.
        threshold (float): Seuil minimal d'amélioration.
        patience (int): Nombre d'époques à patienter sans amélioration.
        count (int): Compteur actuel d'époques sans amélioration.

    Returns:
        tuple: (booléen indiquant l'arrêt, nouveau compteur)
    """
    if opt_cost - cost > threshold:
        # Amélioration significative : réinitialisation du compteur
        return (False, 0)
    else:
        # Aucune amélioration : incrémenter le compteur
        new_count = count + 1
        if new_count >= patience:
            return (True, new_count)
        else:
            return (False, new_count)
