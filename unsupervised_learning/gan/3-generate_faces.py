#!/usr/bin/env python3
"""
Projet GAN : Génération de visages à l'aide de
réseaux générateurs adverses (GAN)

Ce script définit et entraîne un modèle GAN pour générer des visages.
Il inclut la définition des modèles générateur et discriminateur,
ainsi que le processus d'entraînement.

"""

import numpy as np
# Charger les images
import matplotlib.pyplot as plt
import tensorflow.keras as K

def convolutional_GenDiscr():
    # Modèle générateur et modèle discriminateur

    def get_generator():
        # Définir l'entrée du générateur
        inputs = K.Input(shape=(16,))

        # Première couche dense avec activation tanh
        hidden = K.layers.Dense(2048, activation="tanh")(inputs)
        # Reshape pour préparer les données pour les couches de convolution
        hidden = K.layers.Reshape((2, 2, 512))(hidden)
        # Upsampling pour augmenter la taille de l'image
        hidden = K.layers.UpSampling2D((2, 2))(hidden)

        # Première couche de convolution
        hidden = K.layers.Conv2D(filters=64, kernel_size=3, padding="same")(hidden)
        hidden = K.layers.BatchNormalization()(hidden)
        hidden = K.layers.Activation("tanh")(hidden)
        hidden = K.layers.UpSampling2D((2, 2))(hidden)

        # Deuxième couche de convolution
        hidden = K.layers.Conv2D(filters=16, kernel_size=3, padding="same")(hidden)
        hidden = K.layers.BatchNormalization()(hidden)
        hidden = K.layers.Activation("tanh")(hidden)
        hidden = K.layers.UpSampling2D((2, 2))(hidden)

        # Troisième couche de convolution
        hidden = K.layers.Conv2D(filters=1, kernel_size=3, padding="same")(hidden)
        hidden = K.layers.BatchNormalization()(hidden)
        outputs = K.layers.Activation("tanh")(hidden)

        # Créer le modèle générateur
        generator = K.Model(inputs, outputs, name="generator")
        return generator

    def get_discriminator():
        # Définir l'entrée du discriminateur
        inputs = K.Input(shape=(16, 16, 1))

        # Première couche de convolution
        hidden = K.layers.Conv2D(filters=32, kernel_size=3, padding="same")(inputs)
        hidden = K.layers.MaxPooling2D((2, 2))(hidden)
        hidden = K.layers.Activation("tanh")(hidden)

        # Deuxième couche de convolution
        hidden = K.layers.Conv2D(filters=64, kernel_size=3, padding="same")(hidden)
        hidden = K.layers.MaxPooling2D((2, 2))(hidden)
        hidden = K.layers.Activation("tanh")(hidden)

        # Troisième couche de convolution
        hidden = K.layers.Conv2D(filters=128, kernel_size=3, padding="same")(hidden)
        hidden = K.layers.MaxPooling2D((2, 2))(hidden)
        hidden = K.layers.Activation("tanh")(hidden)

        # Quatrième couche de convolution
        hidden = K.layers.Conv2D(filters=256, kernel_size=3, padding="same")(hidden)
        hidden = K.layers.MaxPooling2D((2, 2))(hidden)
        hidden = K.layers.Activation("tanh")(hidden)

        # Aplatir les données pour la couche dense finale
        hidden = K.layers.Flatten()(hidden)
        outputs = K.layers.Dense(1, activation="tanh")(hidden)

        # Créer le modèle discriminateur
        discriminator = K.Model(inputs, outputs, name="discriminator")
        return discriminator

    return get_generator(), get_discriminator()

# Obtenir les modèles générateur et discriminateur
gen, discr = convolutional_GenDiscr()
# Afficher les résumés des modèles
print(gen.summary(line_length=100))
print(discr.summary(line_length=100))
