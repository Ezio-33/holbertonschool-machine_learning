#!/usr/bin/env python3
"""
Module pour sauvegarder et charger les poids d'un modèle Keras
"""
import tensorflow.keras as K


def save_weights(network, filename):
    """
    Sauvegarde les poids d'un modèle Keras dans un fichier
    Args:
        network: modèle Keras dont les poids doivent être sauvegardés
        filename: chemin du fichier où sauvegarder les poids
    Returns:
        None
    """
    if not filename.endswith('.weights.h5'):
        filename = f"{filename}.weights.h5"
    try:
        network.save_weights(filename)
        return None
    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")
        return None


def load_weights(network, filename):
    """
    Charge des poids sauvegardés dans un modèle Keras
    Args:
        network: modèle Keras dans lequel charger les poids
        filename: chemin du fichier contenant les poids à charger
    Returns:
        None
    """
    if not filename.endswith('.weights.h5'):
        filename = f"{filename}.weights.h5"
    try:
        network.load_weights(filename)
        return None
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        return None