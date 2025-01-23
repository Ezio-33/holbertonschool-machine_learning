#!/usr/bin/env python3
"""
Module contenant la fonction de décroissance du taux d'apprentissage
utilisant TensorFlow
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Crée une opération de décroissance du taux d'apprentissage.

    Args:
        alpha: taux d'apprentissage initial
        decay_rate: facteur de décroissance
        decay_step: nombre d'étapes avant chaque décroissance

    Returns:
        Un objet InverseTimeDecay de TensorFlow
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
