#!/usr/bin/env python3
"""
Module pour sauvegarder et charger la configuration d'un modèle Keras
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Sauvegarde la configuration d'un modèle Keras

    Arguments:
        network: le modèle dont la configuration doit être sauvegardée
        filename: chemin du fichier où sauvegarder la configuration

    Returns:
        None
    """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)
    return None


def load_config(filename):
    """
    Charge une configuration et crée un modèle

    Arguments:
        filename: chemin du fichier contenant la configuration

    Returns:
        Le modèle Keras créé à partir de la configuration
    """
    with open(filename, 'r') as f:
        config = f.read()
    return K.models.model_from_json(config)
