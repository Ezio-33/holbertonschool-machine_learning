#!/usr/bin/env python3
"""Implémentation de LeNet-5 avec Keras"""

from tensorflow import keras as K


def lenet5(X):
    """
    Construit LeNet-5 avec Keras

    Args:
        X: K.Input de forme (28, 28, 1) - Entrée des images

    Returns:
        Model Keras compilé
    """
    initializer = K.initializers.VarianceScaling(scale=2.0, seed=0)

    # Couche 1 : Conv2D -> ReLU -> MaxPool
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer
    )(X)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # Couche 2 : Conv2D -> ReLU -> MaxPool
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=initializer
    )(pool1)

    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Aplatissement
    flat = K.layers.Flatten()(pool2)

    # Couche 3 : Dense -> ReLU
    dense1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=initializer
    )(flat)

    # Couche 4 : Dense -> ReLU
    dense2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=initializer
    )(dense1)

    # Sortie finale
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=initializer
    )(dense2)

    # Création et compilation du modèle
    model = K.Model(inputs=X, outputs=output)
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
