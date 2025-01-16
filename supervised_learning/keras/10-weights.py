#!/usr/bin/env python3
"""
Module pour sauvegarder et charger les poids d'un modèle Keras
"""
import tensorflow.keras as K


def save_weights(network, filename):
    """
    Sauvegarde les poids d'un modèle Keras
    """
    if not filename.endswith('.weights.h5'):
        filename = f"{filename}.weights.h5"
    try:
        network.save_weights(filename)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")
    return None


def load_weights(network, filename):
    """
    Charge des poids dans un modèle Keras
    """
    if not filename.endswith('.weights.h5'):
        filename = f"{filename}.weights.h5"
    try:
        network.load_weights(filename)
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
    return None
