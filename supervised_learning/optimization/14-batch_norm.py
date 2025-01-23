#!/usr/bin/env python3
"""Module contenant la fonction pour créer une couche de batch normalization"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Crée une couche de normalisation par lots pour un réseau de neurones.

    Args:
        prev: sortie activée de la couche précédente
        n: nombre de noeuds dans la couche à créer
        activation: fonction d'activation à utiliser sur la sortie de la couche

    Returns:
        La sortie activée pour la couche
    """
    # Initialisation avec VarianceScaling
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Couche dense
    dense_layer = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    Z = dense_layer(prev)

    # Paramètres gamma et beta
    gamma = tf.Variable(initial_value=tf.ones((1, n)), trainable=True)
    beta = tf.Variable(initial_value=tf.zeros((1, n)), trainable=True)

    # Calcul de la moyenne et de la variance
    mean, variance = tf.nn.moments(Z, axes=[0])

    # Normalisation
    epsilon = 1e-7
    Z_norm = tf.nn.batch_normalization(
        Z, mean, variance, beta, gamma, epsilon)

    # Application de l'activation
    return activation(Z_norm)
