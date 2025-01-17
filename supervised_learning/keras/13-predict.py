#!/usr/bin/env python3
"""
Module pour effectuer des prédictions avec un modèle de deep learning
"""
import tensorflow.keras as K


def predict(network, data):
    """
    Effectue des prédictions en utilisant un modèle de deep learning

    Arguments:
        network: modèle Keras entraîné
        data: numpy.ndarray contenant les données d'entrée

    Returns:
        numpy.ndarray contenant les prédictions
    """
    predictions = network.predict(
        x=data,
        verbose=0
    )

    return predictions
