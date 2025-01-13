#!/usr/bin/env python3
"""
Module pour calculer la perte d'entropie croisée softmax
"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Calcule la perte d'entropie croisée softmax des prédictions

    Args:
        y: placeholder pour les étiquettes réelles
        y_pred: tenseur contenant les prédictions du réseau

    Returns:
        tenseur contenant la perte de la prédiction
    """
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y,
        logits=y_pred
    ))
