#!/usr/bin/env python3
"""
Module pour l'entraînement d'un modèle avec early stopping
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Entraîne un modèle avec early stopping

    Arguments:
        network: le modèle à entraîner
        data: numpy.ndarray contenant les données d'entrée
        labels: numpy.ndarray one-hot contenant les étiquettes
        batch_size: taille des lots pour la descente de gradient
        epochs: nombre d'itérations d'entraînement
        validation_data: données pour la validation (tuple)
        early_stopping: booléen pour l'arrêt précoce
        patience: nombre d'époques à attendre avant l'arrêt
        verbose: booléen pour l'affichage de la progression
        shuffle: booléen pour mélanger les données

    Returns:
        L'historique de l'entraînement
    """
    callbacks = []

    if validation_data and early_stopping:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )

    return history
