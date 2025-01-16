#!/usr/bin/env python3
"""
Module pour sauvegarder et charger des modèles Keras complets
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Sauvegarde un modèle Keras complet

    Arguments:
        network: le modèle à sauvegarder
        filename: chemin du fichier où sauvegarder le modèle

    Returns:
        None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Charge un modèle Keras complet

    Arguments:
        filename: chemin du fichier contenant le modèle à charger

    Returns:
        Le modèle Keras chargé
    """
    return K.models.load_model(filename)
