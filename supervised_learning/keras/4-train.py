#!/usr/bin/env python3
"""
Module pour l'entraînement d'un modèle de réseau neuronal avec Keras
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Entraîne un modèle de réseau neuronal

    Arguments:
        network: le modèle à entraîner
        data: numpy.ndarray contenant les données d'entrée
        labels: numpy.ndarray one-hot contenant les étiquettes
        batch_size: taille du batch pour la descente de gradient mini-batch
        epochs: nombre d'itérations d'entraînement
        verbose: booléen pour l'affichage de la progression
        shuffle: booléen pour le mélange des batches

    Returns:
        L'historique de l'entraînement
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
