#!/usr/bin/env python3
"""
Module pour construire un réseau de neurones avec l'API fonctionnelle de Keras
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Construit un réseau de neurones en utilisant l'API fonctionnelle Keras

    Arguments:
        nx: nombre de features en entrée
        layers: liste contenant le nombre de nœuds pour chaque couche
        activations: liste des fonctions d'activation pour chaque couche
        lambtha: paramètre de régularisation L2
        keep_prob: probabilité de conservation pour le dropout

    Returns:
        Le modèle Keras compilé
    """
    # Création de la couche d'entrée
    inputs = K.Input(shape=(nx,))
    x = inputs

    # Construction des couches
    for i in range(len(layers)):
        # Couche dense avec régularisation L2
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        # Ajout du dropout sauf pour la dernière couche
        if i != len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    # Création du modèle
    model = K.Model(inputs=inputs, outputs=x)

    return model
