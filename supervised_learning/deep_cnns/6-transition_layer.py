#!/usr/bin/env python3
"""Module implémentant une couche de transition pour DenseNet"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Construit une couche de transition pour DenseNet-C

    Args:
        X: Entrée de la couche
        nb_filters: Nombre de filtres actuels
        compression: Taux de réduction des filtres (0-1)

    Returns:
        Tuple (sortie de la couche, nouveau nombre de filtres)
    """
    # Calcul des nouveaux filtres avec compression
    new_filters = int(nb_filters * compression)

    # Initialisation He Normal avec seed=0
    initializer = K.initializers.HeNormal(seed=0)

    # BatchNorm -> ReLU
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    # Conv 1x1 pour réduire les canaux
    X = K.layers.Conv2D(new_filters, (1, 1), padding='same',
                        kernel_initializer=initializer)(X)

    # Average Pooling 2x2 avec stride 2
    X = K.layers.AveragePooling2D((2, 2), strides=2)(X)

    return X, new_filters
