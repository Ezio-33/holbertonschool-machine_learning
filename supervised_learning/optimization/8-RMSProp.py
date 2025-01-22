#!/usr/bin/env python3
"""
Module contenant la fonction de création d'un optimiseur RMSProp
utilisant TensorFlow
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Crée un optimiseur RMSProp.

    Args:
        alpha: taux d'apprentissage
        beta2: facteur de décroissance pour la moyenne mobile
        epsilon: petit nombre pour éviter la division par zéro

    Returns:
        Un optimiseur TensorFlow configuré avec RMSProp
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
