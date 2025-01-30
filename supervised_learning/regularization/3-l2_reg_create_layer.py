#!/usr/bin/env python3
"""
Crée une couche de réseau neuronal avec régularisation L2
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Crée une couche Dense avec régularisation L2

    Args:
        prev: Tenseur de sortie de la couche précédente
        n: Nombre de nœuds dans la couche à créer
        activation: Fonction d'activation
        lambtha: Paramètre de régularisation L2

    Returns:
        Tenseur de sortie de la nouvelle couche
    """
    # Initialisation des poids avec VarianceScaling
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg")

    # Régularisateur L2
    regularizer = tf.keras.regularizers.L2(lambtha)

    # Création de la couche Dense
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )

    return layer(prev)
