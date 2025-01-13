#!/usr/bin/env python3
"""
Module pour calculer la précision des prédictions d'un réseau neuronal
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Calcule la précision des prédictions d'un réseau neuronal

    Args:
        y: placeholder pour les étiquettes réelles
        y_pred: tenseur contenant les prédictions du réseau

    Returns:
        tenseur contenant la précision décimale des prédictions
    """
    # Obtenir les indices des valeurs maximales
    correct_predictions = tf.equal(
        tf.argmax(y, 1),
        tf.argmax(y_pred, 1)
    )

    # Calculer la moyenne des prédictions correctes
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy
