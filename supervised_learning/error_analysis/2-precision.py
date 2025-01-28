#!/usr/bin/env python3
import numpy as np


def precision(confusion):
    """
    Calcule la précision pour chaque classe à partir
    d'une matrice de confusion.

    Args:
        confusion: numpy.ndarray de forme (classes, classes)
                  où les indices des lignes représentent les vraies étiquettes
                  et les indices des colonnes représentent les prédictions

    Returns:
        numpy.ndarray de forme (classes,) contenant
        la précision de chaque classe
    """
    # Obtenir les vrais positifs (diagonale de la matrice)
    true_positives = np.diag(confusion)

    # Calculer le total des prédictions positives (somme des colonnes)
    predicted_positives = np.sum(confusion, axis=0)

    # Éviter la division par zéro
    predicted_positives = np.where(
        predicted_positives == 0, 1, predicted_positives)

    # Calculer la précision pour chaque classe
    precision_scores = true_positives / predicted_positives

    return precision_scores
