#!/usr/bin/env python3
"""
Module implémentant un bloc d'identité ResNet comme décrit dans
'Deep Residual Learning for Image Recognition (2015)'
"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Construit un bloc d'identité pour ResNet

    Args:
        A_prev: Sortie de la couche précédente
        filters: Tuple contenant F11, F3, F12
            F11: Nombre de filtres de la première convolution 1x1
            F3: Nombre de filtres de la convolution 3x3
            F12: Nombre de filtres de la seconde convolution 1x1

    Returns:
        Sortie activée du bloc d'identité
    """
    # Extraction des paramètres
    F11, F3, F12 = filters

    # Initialisation He normal avec seed=0
    initializer = K.initializers.HeNormal(seed=0)

    # Première convolution 1x1
    conv1 = K.layers.Conv2D(F11, (1, 1),
                            kernel_initializer=initializer,
                            padding='same')(A_prev)
    batch1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(batch1)

    # Convolution 3x3
    conv2 = K.layers.Conv2D(F3, (3, 3),
                            kernel_initializer=initializer,
                            padding='same')(act1)
    batch2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(batch2)

    # Seconde convolution 1x1
    conv3 = K.layers.Conv2D(F12, (1, 1),
                            kernel_initializer=initializer,
                            padding='same')(act2)
    batch3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Addition avec connexion résiduelle
    add = K.layers.Add()([batch3, A_prev])

    # Activation finale
    return K.layers.Activation('relu')(add)
