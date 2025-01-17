#!/usr/bin/env python3
"""
Module pour tester un modèle de deep learning
"""
import tensorflow.keras as K


def test_model(network, data, labels):
    """
    Teste un modèle de deep learning

    Args:
        network: modèle à tester
        data: données d'entrée de test
        labels: étiquettes des données de test

    Returns:
        float: perte et précision du modèle
    """
    return network.evaluate(data, labels, verbose=0)
