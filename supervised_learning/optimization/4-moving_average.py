#!/usr/bin/env python3
"""
Module contenant la fonction de calcul de la moyenne mobile
"""
import numpy as np


def moving_average(data, beta):
    """
    Calcule la moyenne mobile pondérée d'un ensemble de données.

    Args:
        data: liste des données à moyenner
        beta: facteur de pondération pour la moyenne mobile

    Returns:
        Une liste contenant les moyennes mobiles des données
    """
    v = 0
    moving_averages = []
    
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        # Correction du biais
        correction = 1 - beta ** (i + 1)
        moving_averages.append(v / correction)
    
    return moving_averages
