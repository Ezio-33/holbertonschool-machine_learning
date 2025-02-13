#!/usr/bin/env python3
"""Module implémentant un Dense Block pour DenseNet"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Construit un Dense Block selon l'architecture DenseNet-B

    Args:
        X: Entrée du bloc
        nb_filters: Nombre de filtres initiaux
        growth_rate: Taux de croissance des filtres
        layers: Nombre de couches dans le bloc

    Returns:
        Tuple (sortie concaténée, nombre total de filtres)
    """
    init = K.initializers.HeNormal(seed=0)

    for _ in range(layers):
        # Bottleneck layer
        X_copy = X

        # BN -> ReLU -> Conv1x1
        X = K.layers.BatchNormalization()(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(4 * growth_rate, (1, 1),
                            kernel_initializer=init)(X)

        # BN -> ReLU -> Conv3x3
        X = K.layers.BatchNormalization()(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                            kernel_initializer=init)(X)

        # Concaténation avec l'entrée originale
        X = K.layers.concatenate([X_copy, X])
        nb_filters += growth_rate

    return X, nb_filters
