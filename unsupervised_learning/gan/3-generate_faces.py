#!/usr/bin/env python3
"""
Générateur et discriminateur convolutionnels pour la génération de visages.
"""
import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """
    Construit les modèles générateur et discriminateur en
    utilisant des couches convolutionnelles.

    Returns:
        generator (tf.keras.Model): Le modèle générateur.
        discriminator (tf.keras.Model): Le modèle discriminateur.
    """

    def generator():
        """
        Construit le modèle générateur.
        """
        model = keras.Sequential([
            keras.layers.Input(shape=(16,)),
            keras.layers.Dense(2048),
            keras.layers.Reshape((2, 2, 512)),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2D(64, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('tanh'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2D(16, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('tanh'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2D(1, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('tanh')
        ], name='generator')
        return model

    def discriminator():
        """
        Construit le modèle discriminateur.
        """
        model = keras.Sequential([
            keras.layers.Input(shape=(16, 16, 1)),
            keras.layers.Conv2D(32, (3, 3), padding='same'),
            keras.layers.MaxPooling2D(),
            keras.layers.Activation('tanh'),
            keras.layers.Conv2D(64, (3, 3), padding='same'),
            keras.layers.MaxPooling2D(),
            keras.layers.Activation('tanh'),
            keras.layers.Conv2D(128, (3, 3), padding='same'),
            keras.layers.MaxPooling2D(),
            keras.layers.Activation('tanh'),
            keras.layers.Conv2D(256, (3, 3), padding='same'),
            keras.layers.MaxPooling2D(),
            keras.layers.Activation('tanh'),
            keras.layers.Flatten(),
            keras.layers.Dense(1)
        ], name='discriminator')
        return model

    return generator(), discriminator()
