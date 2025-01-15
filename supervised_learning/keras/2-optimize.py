#!/usr/bin/env python3
"""
Module pour optimiser un modèle Keras avec l'optimiseur Adam
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Configure l'optimisation d'un modèle Keras avec Adam

    Arguments:
        network: le modèle à optimiser
        alpha: le taux d'apprentissage
        beta1: premier paramètre de l'optimisation Adam
        beta2: second paramètre de l'optimisation Adam

    Returns:
        None
    """
    # Création de l'optimiseur Adam avec les paramètres spécifiés
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    # Compilation du modèle
    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
