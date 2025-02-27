#!/usr/bin/env python3
"""Module implémentant le réseau Inception complet"""

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Construit le réseau Inception selon l'architecture originale
    Returns:
        Model Keras compilé
    """
    # Initialisation He Normal avec seed=0
    initializer = K.initializers.HeNormal(seed=0)

    # Couche d'entrée (conserve la référence originale)
    input_layer = K.Input(shape=(224, 224, 3))

    # Partie initiale
    x = K.layers.Conv2D(64, (7, 7),
                        strides=(2, 2),
                        padding='same',
                        activation='relu',
                        kernel_initializer=initializer)(input_layer)

    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Conv 3x3
    x = K.layers.Conv2D(64, (1, 1),
                        padding='same',
                        activation='relu',
                        kernel_initializer=initializer)(x)

    x = K.layers.Conv2D(192, (3, 3),
                        padding='same',
                        activation='relu',
                        kernel_initializer=initializer)(x)

    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Blocs Inception (9 blocs au total)
    x = inception_block(x, [64, 96, 128, 16, 32, 32])  # 3a
    x = inception_block(x, [128, 128, 192, 32, 96, 64])  # 3b
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_block(x, [192, 96, 208, 16, 48, 64])  # 4a
    x = inception_block(x, [160, 112, 224, 24, 64, 64])  # 4b
    x = inception_block(x, [128, 128, 256, 24, 64, 64])  # 4c
    x = inception_block(x, [112, 144, 288, 32, 64, 64])  # 4d
    x = inception_block(x, [256, 160, 320, 32, 128, 128])  # 4e

    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_block(x, [256, 160, 320, 32, 128, 128])  # 5a
    x = inception_block(x, [384, 192, 384, 48, 128, 128])  # 5b

    # Couches finales
    x = K.layers.AveragePooling2D((7, 7))(x)
    x = K.layers.Dropout(0.4)(x)
    output_layer = K.layers.Dense(1000,
                                  activation='softmax',
                                  kernel_initializer=initializer)(x)

    return K.models.Model(inputs=input_layer, outputs=output_layer)
