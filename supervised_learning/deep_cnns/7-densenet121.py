#!/usr/bin/env python3
"""Module implémentant DenseNet-121"""

from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Construit DenseNet-121 selon l'architecture originale

    Args:
        growth_rate: Taux de croissance des filtres
        compression: Facteur de compression pour les couches de transition

    Returns:
        Le modèle Keras
    """
    # Initialisation
    kernel_init = K.initializers.he_normal(seed=0)

    # Couche d'entrée
    X = K.Input(shape=(224, 224, 3))

    # Pré-traitement
    batchNorm0 = K.layers.BatchNormalization(axis=3)(X)
    activation0 = K.layers.Activation('relu')(batchNorm0)

    # Première convolution
    layer1 = K.layers.Conv2D(
        filters=2 * growth_rate,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=kernel_init
    )(activation0)

    # Max pooling initial
    l1pool = K.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(layer1)

    # Dense block 1 (6 couches) + transition
    layer2, n_f2 = dense_block(l1pool, 2 * growth_rate, growth_rate, 6)
    layer3, n_f3 = transition_layer(layer2, n_f2, compression)

    # Dense block 2 (12 couches) + transition
    layer4, n_f4 = dense_block(layer3, n_f3, growth_rate, 12)
    layer5, n_f5 = transition_layer(layer4, n_f4, compression)

    # Dense block 3 (24 couches) + transition
    layer6, n_f6 = dense_block(layer5, n_f5, growth_rate, 24)
    layer7, n_f7 = transition_layer(layer6, n_f6, compression)

    # Dense block 4 (16 couches)
    layer8, n_f8 = dense_block(layer7, n_f7, growth_rate, 16)

    # Classification finale
    avg_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=7,
        padding='same'
    )(layer8)

    # Couche Dense finale
    Y = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=kernel_init
    )(avg_pool)

    # Création du modèle
    model = K.models.Model(inputs=X, outputs=Y)

    return model
