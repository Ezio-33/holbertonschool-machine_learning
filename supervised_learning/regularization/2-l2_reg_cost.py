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
        model: modèle Keras qui inclut des couches avec régularisation L2

    Returns:
        tensor contenant le coût total avec régularisation L2
    """
    # Additionne le coût initial et toutes les pertes de régularisation du modèle
    l2_cost = cost
    
    # Parcours toutes les couches du modèle
    for layer in model.layers:
        # Ajoute les pertes de régularisation de chaque couche si elles existent  
        if len(layer.losses) > 0:
            l2_cost += tf.math.add_n(layer.losses)
    
    return l2_cost
