#!/usr/bin/env python3
"""
Module pour créer une couche de réseau neuronal avec Dropout
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Crée une couche de réseau neuronal avec régularisation Dropout.

    Args:
        prev: tenseur contenant la sortie de la couche précédente
        n: nombre de nœuds dans la nouvelle couche
        activation: fonction d'activation à utiliser
        keep_prob: probabilité de garder un neurone
        training: booléen indiquant si le modèle est en entraînement

    Returns:
        tenseur de sortie de la nouvelle couche
    """
    # Initialisation des poids avec la méthode spécifiée
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")

    # Création de la couche Dense
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )

    # Application de la couche Dense
    output = layer(prev)

    # Ajout de la couche Dropout
    if keep_prob < 1:
        output = tf.keras.layers.Dropout(
            rate=1 -
            keep_prob)(
            output,
            training=training)

    return output
