#!/usr/bin/env python3
"""
Module contenant la fonction forward_prop pour créer
le graphe de propagation avant d'un réseau neuronal
"""
import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Crée le graphe de propagation avant pour le réseau neuronal

    Args:
        x: placeholder pour les données d'entrée
        layer_sizes: liste contenant le nombre de nœuds dans chaque couche
        activations: liste des fonctions d'activation pour chaque couche

    Returns:
        prédiction du réseau sous forme de tenseur
    """
    create_layer = __import__('1-create_layer').create_layer

    layer_output = x
    for i in range(len(layer_sizes)):
        layer_output = create_layer(
            layer_output,
            layer_sizes[i],
            activations[i]
        )
    return layer_output
