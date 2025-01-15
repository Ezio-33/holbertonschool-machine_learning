#!/usr/bin/env python3
"""
Module pour l'entraînement d'un modèle avec validation
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Entraîne un modèle de réseau neuronal et valide ses performances

    Arguments:
        network: le modèle à entraîner
        data: numpy.ndarray contenant les données d'entrée
        labels: numpy.ndarray one-hot contenant les étiquettes
        batch_size: taille des lots pour la descente de gradient
        epochs: nombre d'itérations d'entraînement
        validation_data: données pour la validation (tuple)
        verbose: booléen déterminant l'affichage de la progression
        shuffle: booléen pour mélanger les données à chaque époque

    Returns:
        L'historique de l'entraînement
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data
    )

    return history
