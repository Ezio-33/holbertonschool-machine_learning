#!/usr/bin/env python3
"""Module implémentant un bloc Inception comme décrit dans 'Going Deeper
with Convolutions' (2014)"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Construit un bloc Inception selon l'architecture originale

    Args:
        A_prev: Couche précédente du réseau
        filters: Tuple (F1, F3R, F3, F5R, F5, FPP) contenant :
            F1 - Filtres 1x1
            F3R - Filtres réducteurs avant 3x3
            F3 - Filtres 3x3
            F5R - Filtres réducteurs avant 5x5
            F5 - Filtres 5x5
            FPP - Filtres post-pooling

    Returns:
        Sortie concaténée du bloc Inception
    """
    # Récupération des paramètres
    F1, F3R, F3, F5R, F5, FPP = filters

    # Branche 1x1
    branch1 = K.layers.Conv2D(
        F1, (1, 1), padding='same', activation='relu')(A_prev)

    # Branche 3x3 avec réduction
    branch3 = K.layers.Conv2D(
        F3R, (1, 1), padding='same', activation='relu')(A_prev)
    branch3 = K.layers.Conv2D(
        F3, (3, 3), padding='same', activation='relu')(branch3)

    # Branche 5x5 avec réduction
    branch5 = K.layers.Conv2D(
        F5R, (1, 1), padding='same', activation='relu')(A_prev)
    branch5 = K.layers.Conv2D(
        F5, (5, 5), padding='same', activation='relu')(branch5)

    # Branche MaxPooling
    branch_pool = K.layers.MaxPooling2D(
        (3, 3), strides=(1, 1), padding='same')(A_prev)
    branch_pool = K.layers.Conv2D(
        FPP, (1, 1), padding='same', activation='relu')(branch_pool)

    # Concaténation des branches
    return K.layers.concatenate([branch1, branch3, branch5, branch_pool])
