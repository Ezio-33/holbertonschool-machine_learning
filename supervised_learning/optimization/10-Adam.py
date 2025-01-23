#!/usr/bin/env python3
"""
Module contenant la fonction de création d'un optimiseur Adam
utilisant TensorFlow
"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Crée un optimiseur Adam.

    Args:
        alpha: taux d'apprentissage
        beta1: facteur pour la moyenne mobile du premier moment
        beta2: facteur pour la moyenne mobile du second moment
        epsilon: petit nombre pour éviter la division par zéro

    Returns:
        Un optimiseur TensorFlow configuré avec Adam
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
