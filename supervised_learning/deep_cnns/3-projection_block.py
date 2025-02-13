#!/usr/bin/env python3
"""Implémentation d'un bloc de projection pour ResNet"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Construit un bloc de projection selon l'architecture ResNet

    Args:
        A_prev: Sortie de la couche précédente
        filters: Tuple (F11, F3, F12) spécifiant les filtres
        s: Stride pour la première convolution (défaut 2)

    Returns:
        Tenseur de sortie activé du bloc
    """
    # Initialisation des paramètres
    F11, F3, F12 = filters

    # Initialisation He Normal avec seed=0
    initializer = K.initializers.HeNormal(seed=0)

    # Chemin principal
    # Première couche 1x1
    conv1 = K.layers.Conv2D(F11, (1, 1), strides=(s, s),
                            kernel_initializer=initializer,
                            padding='valid')(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    # Couche 3x3
    conv2 = K.layers.Conv2D(F3, (3, 3), padding='same',
                            kernel_initializer=initializer)(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    # Couche 1x1 finale
    conv3 = K.layers.Conv2D(F12, (1, 1), padding='valid',
                            kernel_initializer=initializer)(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Chemin raccourci
    shortcut = K.layers.Conv2D(F12, (1, 1), strides=(s, s),
                               kernel_initializer=initializer,
                               padding='valid')(A_prev)
    shortcut_bn = K.layers.BatchNormalization(axis=3)(shortcut)

    # Fusion des chemins
    add = K.layers.Add()([bn3, shortcut_bn])

    return K.layers.Activation('relu')(add)
