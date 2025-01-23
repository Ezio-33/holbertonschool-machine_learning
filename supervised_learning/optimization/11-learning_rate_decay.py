#!/usr/bin/env python3
"""
Module contenant la fonction de décroissance du taux d'apprentissage
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Met à jour le taux d'apprentissage en utilisant la
    décroissance inverse avec le temps.

    Args:
        alpha: taux d'apprentissage initial
        decay_rate: poids utilisé pour déterminer le taux de décroissance
        global_step: nombre d'itérations effectuées
        decay_step: nombre d'itérations avant chaque décroissance

    Returns:
        Le taux d'apprentissage mis à jour
    """
    # Calcul du nombre de fois que le taux a été décru
    step = np.floor(global_step / decay_step)
    
    # Application de la formule de décroissance inverse avec le temps
    return alpha / (1 + decay_rate * step)
