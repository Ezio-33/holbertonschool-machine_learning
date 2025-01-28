#!/usr/bin/env python3
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Crée une matrice de confusion à partir des étiquettes réelles
    et prédites.

    Args:
        labels (numpy.ndarray): de forme (m, classes) contenant
        les étiquettes réelles en format one-hot
        logits (numpy.ndarray): de forme (m, classes) contenant les prédictions
               en format one-hot

    Returns:
        numpy.ndarray: de forme (classes, classes) représentant
        la matrice de confusion où les lignes correspondent aux
        classes réelles et les colonnes aux prédictions
    """
    # Convertir les matrices one-hot en indices de classe
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    # Obtenir le nombre de classes
    n_classes = labels.shape[1]

    # Initialiser la matrice de confusion avec des zéros
    confusion = np.zeros((n_classes, n_classes))

    # Remplir la matrice de confusion
    for true, pred in zip(true_classes, pred_classes):
        confusion[true][pred] += 1

    return confusion
