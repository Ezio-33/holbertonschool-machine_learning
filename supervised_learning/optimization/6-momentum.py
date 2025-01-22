#!/usr/bin/env python3
"""
Module contenant la fonction de création d'un optimiseur avec momentum
utilisant TensorFlow
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Crée un optimiseur de descente de gradient avec momentum.

    Args:
        alpha: taux d'apprentissage (learning rate)
        beta1: facteur de momentum

    Returns:
        Un optimiseur TensorFlow configuré avec momentum
    """
    return tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )
