#!/usr/bin/env python3
"""
Calcule le coût d'un réseau de neurones avec régularisation L2,
en retournant un tenseur contenant le coût pour chaque couche.
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calcule le coût total, couche par couche, incluant la régularisation L2

    Arguments:
        cost: tensor contenant le coût du réseau sans régularisation L2
        model: un modèle Keras qui inclut des couches avec régularisation L2

    Returns:
        tensor de forme (nb_couches,) contenant le coût total
        pour chacune des couches, incluant la régularisation L2
    """
    # Nous construisons un vecteur, chaque entrée correspondant au coût + pertes
    # de régularisation de la couche correspondante
    costs = []
    for layer in model.layers:
        # Si la couche ne possède pas de pénalités de régularisation,
        # elle n'ajoute rien au coût
        if len(layer.losses) > 0:
            # On ajoute au coût initial la somme des régularisations de la couche
            costs.append(cost + tf.math.add_n(layer.losses))
        else:
            # Sinon, c'est juste le coût de base
            costs.append(cost)

    # Empile chaque coût en un seul tenseur 1D
    return tf.stack(costs)
