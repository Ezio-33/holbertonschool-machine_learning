#!/usr/bin/env python3
"""
Module pour construire un réseau de neurones avec Keras
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Construit un réseau de neurones avec Keras

    Arguments:
        nx: nombre de features en entrée
        layers: liste contenant le nombre de nœuds pour chaque couche
        activations: liste des fonctions d'activation pour chaque couche
        lambtha: paramètre de régularisation L2
        keep_prob: probabilité de conservation pour le dropout

    Returns:
        Le modèle Keras compilé
    """
    # Initialisation du modèle séquentiel
    model = K.Sequential()

    # Construction des couches
    for i in range(len(layers)):
        # Première couche
        if i == 0:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_shape=(nx,)
            ))
        # Couches suivantes
        else:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)
            ))

        # Ajout du dropout sauf pour la dernière couche
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
