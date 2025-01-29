#!/usr/bin/env python3
"""
Calcule le coût d'un réseau de neurones avec régularisation L2
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calcule le coût total incluant la régularisation L2

    Arguments:
        cost: tensor contenant le coût du réseau sans régularisation L2
        model: un modèle Keras qui inclut des couches avec régularisation L2

    Returns:
        tensor contenant le coût total pour chaque couche du réseau,
        incluant la régularisation L2
    """
    # Récupérer les pertes de régularisation pour chaque couche
    reg_losses = [tf.reduce_sum(layer.losses) for layer in model.layers if layer.losses]

    # Si des pertes existent, les ajouter au coût initial
    if reg_losses:
        total_reg_loss = tf.stack(reg_losses)
        return tf.concat([cost[None], total_reg_loss], axis=0)

    # Si aucune perte, retourner simplement le coût initial
    return cost
