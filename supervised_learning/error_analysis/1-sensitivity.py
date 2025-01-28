#!/usr/bin/env python3
"""
Calcule la sensibilité pour chaque classe à partir
d'une matrice de confusion.
"""

import numpy as np


def sensitivity(confusion):
    """
    Args:
        confusion (numpy.ndarray): de forme (classes, classes)
                  où les indices des lignes représentent les vraies étiquettes
                  et les indices des colonnes représentent les prédictions

    Returns:
        numpy.ndarray: de forme (classes,) contenant la
        sensibilité de chaque classe
    """
    # Obtenir les vrais positifs (diagonale de la matrice)
    true_positives = np.diag(confusion)

    # Calculer le total des cas positifs (somme des lignes)
    total_positives = np.sum(confusion, axis=1)

    # Éviter la division par zéro
    total_positives = np.where(total_positives == 0, 1, total_positives)

    # Calculer la sensibilité
    sensitivity_scores = true_positives / total_positives

    return sensitivity_scores
