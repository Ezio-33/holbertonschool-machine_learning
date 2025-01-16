#!/usr/bin/env python3
"""
Module pour sauvegarder et charger les poids d'un modèle Keras
"""
import tensorflow.keras as K


def save_weights(network, filename):
    """
    Sauvegarde les poids d'un modèle Keras

    Args:
        network: modèle Keras dont les poids doivent être sauvegardés
        filename: chemin du fichier où sauvegarder les poids

    Returns:
        None
    """
    network.save_weights(filename, save_format='h5')
    return None


def load_weights(network, filename):
    """
    Charge des poids dans un modèle Keras

    Args:
        network: modèle Keras dans lequel charger les poids
        filename: chemin du fichier contenant les poids

    Returns:
        None
    """
    network.load_weights(filename)
    return None
