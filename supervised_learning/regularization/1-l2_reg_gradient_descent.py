#!/usr/bin/env python3
"""
L2 Regularization Cost.
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calcule le coût d'un réseau de neurones avec régularisation L2.

    Args:
        cost (float): Le coût initial du réseau de neurones
        sans régularisation.
        model (object): Le modèle contenant les pertes de régularisation L2.

    Returns:
        float: Le coût du réseau de neurones incluant la régularisation L2.
    """
    return cost + model.losses
