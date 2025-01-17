#!/usr/bin/env python3
"""
Module pour tester un modèle de deep learning
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Teste un modèle de deep learning et retourne sa perte et sa précision

    Args:
        network: modèle à tester
        data: données d'entrée de test
        labels: étiquettes des données de test

    Returns:
        list: [perte, précision] du modèle sur les données de test
    """
    return network.evaluate(data, labels, verbose=0)
