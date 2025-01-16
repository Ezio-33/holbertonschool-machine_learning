#!/usr/bin/env python3
"""
Module pour sauvegarder et charger les poids d'un modèle Keras
"""
import tensorflow.keras as K


def save_weights(network, filename):
    """
    Sauvegarde les poids d'un modèle Keras

    Arguments:
        network: le modèle dont les poids doivent être sauvegardés
        filename: chemin du fichier où sauvegarder les poids

    Returns:
        None
    """
    # Ajout de l'extension .weights.h5
    if not filename.endswith('.weights.h5'):
        filename = filename + '.weights.h5'
    network.save_weights(filename)
    return None


def load_weights(network, filename):
    """
    Charge des poids dans un modèle Keras

    Arguments:
        network: le modèle dans lequel charger les poids
        filename: chemin du fichier contenant les poids à charger

    Returns:
        None
    """
    # Ajout de l'extension .weights.h5
    if not filename.endswith('.weights.h5'):
        filename = filename + '.weights.h5'
    network.load_weights(filename)
    return None
