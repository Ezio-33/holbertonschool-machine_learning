#!/usr/bin/env python3
"""
Coût de régularisation L2
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Fonction qui calcule le coût d'un réseau de neurones avec
    régularisation L2
    Arguments:
     - cost est un tenseur contenant le coût du réseau sans
        régularisation L2
    Renvoie:
    Un tenseur contenant le coût du réseau en tenant compte de
    la régularisation L2
    """
    L2_cost = cost + tf.losses.get_regularization_losses()

    return L2_cost