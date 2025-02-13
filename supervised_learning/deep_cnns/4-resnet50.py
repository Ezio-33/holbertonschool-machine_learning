#!/usr/bin/env python3
"""Implémentation de ResNet-50"""

from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Construit l'architecture ResNet-50"""
    init = K.initializers.HeNormal(seed=0)
    input_layer = K.Input(shape=(224, 224, 3))

    # Couche initiale
    x = K.layers.Conv2D(64, (7, 7), strides=2, padding='same',
                        kernel_initializer=init)(input_layer)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Étage 1 (3 blocs)
    x = projection_block(x, [64, 64, 256], s=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    # Étage 2 (4 blocs)
    x = projection_block(x, [128, 128, 512])
    for _ in range(3):
        x = identity_block(x, [128, 128, 512])

    # Étage 3 (6 blocs)
    x = projection_block(x, [256, 256, 1024])
    for _ in range(5):
        x = identity_block(x, [256, 256, 1024])

    # Étage 4 (3 blocs)
    x = projection_block(x, [512, 512, 2048])
    for _ in range(2):
        x = identity_block(x, [512, 512, 2048])

    # Couches finales
    x = K.layers.GlobalAveragePooling2D()(x)
    output = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=init)(x)

    return K.models.Model(inputs=input_layer, outputs=output)
