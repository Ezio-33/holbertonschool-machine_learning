#!/usr/bin/env python3
"""
Module pour tester un modèle de deep learning
"""
import tensorflow.keras as K


def test_model(network, data, labels):
    """
    Teste un modèle de deep learning

    Arguments:
        network: modèle à tester
        data: numpy.ndarray contenant les données de test
        labels: numpy.ndarray one-hot contenant les étiquettes de test

    Returns:
        Les performances du modèle sur les données de test
    """
    evaluation = network.evaluate(
        x=data,
        y=labels,
        verbose=0
    )

    # Conversion explicite en float pour assurer le format exact
    return [float(evaluation[0]), float(evaluation[1])]
