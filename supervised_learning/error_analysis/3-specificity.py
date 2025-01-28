#!/usr/bin/env python3
import numpy as np


def specificity(confusion):
    """
    Calcule la spécificité pour chaque classe à partir
    d'une matrice de confusion.

    Args:
        confusion: numpy.ndarray de forme (classes, classes)
                  où les indices des lignes représentent les vraies étiquettes
                  et les indices des colonnes représentent les prédictions

    Returns:
        numpy.ndarray de forme (classes,) contenant la
        spécificité de chaque classe
    """
    # Obtenir le nombre de classes
    classes = confusion.shape[0]

    # Initialiser le tableau des spécificités
    specificities = np.zeros(classes)

    # Calculer la spécificité pour chaque classe
    for i in range(classes):
        # Vrais négatifs : somme de tous les éléments sauf ceux de la ligne i
        # et colonne i
        true_negatives = np.sum(confusion) - np.sum(confusion[i, :]) - \
            np.sum(confusion[:, i]) + confusion[i, i]

        # Faux positifs : somme de la colonne i moins l'élément diagonal
        false_positives = np.sum(confusion[:, i]) - confusion[i, i]

        # Calculer la spécificité
        specificities[i] = true_negatives / (true_negatives + false_positives)

    return specificities
