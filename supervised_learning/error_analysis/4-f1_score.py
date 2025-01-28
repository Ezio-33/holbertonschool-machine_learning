#!/usr/bin/env python3
import numpy as np


def f1_score(confusion):
    """
    Calcule le score F1 pour chaque classe à partir d'une
    matrice de confusion.

    Args:
        confusion: numpy.ndarray de forme (classes, classes)
                  où les indices des lignes représentent les vraies étiquettes
                  et les indices des colonnes représentent les prédictions

    Returns:
        numpy.ndarray de forme (classes,) contenant
        le score F1 de chaque classe
    """
    # Importer les fonctions précédentes
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    # Calculer la sensibilité (rappel) et la précision
    recall = sensitivity(confusion)
    prec = precision(confusion)

    # Calculer le score F1
    f1 = 2 * (prec * recall) / (prec + recall)

    return f1
