#!/usr/bin/env python3
"""
Module pour créer l'opération d'entraînement d'un réseau neuronal
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Crée l'opération d'entraînement pour le réseau neuronal

    Args:
        loss: tenseur de la perte du réseau
        alpha: taux d'apprentissage

    Returns:
        opération d'entraînement qui utilise la descente de gradient
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op
