#!/usr/bin/env python3
"""
Module pour la conversion d'étiquettes en matrice one-hot
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Convertit un vecteur d'étiquettes en matrice one-hot

    Args:
        labels: vecteur d'étiquettes à convertir
        classes: nombre de classes (si None, utilise max(labels) + 1)

    Returns:
        matrice one-hot où chaque ligne correspond à une étiquette
    """
    # Si classes n'est pas spécifié, utiliser la valeur max + 1
    if classes is None:
        classes = max(labels) + 1

    # Conversion en matrice one-hot avec Keras
    return K.utils.to_categorical(labels, num_classes=classes)
