#!/usr/bin/env python3
"""
Module pour créer une couche de réseau neuronal avec TensorFlow
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Crée une couche du réseau neuronal avec initialisation He et al.
    
    Args:
        prev: tenseur de sortie de la couche précédente
        n: nombre de nœuds dans la couche
        activation: fonction d'activation à utiliser
    
    Returns:
        tenseur de sortie de la couche
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    
    with tf.keras.backend.name_scope('layer'):
        layer = tf.keras.layers.Dense(
            units=n,
            activation=activation,
            kernel_initializer=initializer,
            name='layer'
        )
        return layer(prev)
