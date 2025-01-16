#!/usr/bin/env python3
"""
Module pour sauvegarder et charger la configuration d'un modèle Keras
"""
import tensorflow.keras as K
import json


def save_config(network, filename):
    """
    Sauvegarde la configuration d'un modèle Keras

    Arguments:
        network: le modèle dont la configuration doit être sauvegardée
        filename: chemin du fichier où sauvegarder la configuration

    Returns:
        None
    """
    config = network.get_config()
    with open(filename, 'w') as f:
        json.dump(config, f)
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
        config = json.load(f)
    return K.Model.from_config(config)
