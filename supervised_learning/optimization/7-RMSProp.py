#!/usr/bin/env python3
"""
Module contenant la fonction d'optimisation RMSProp
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Met à jour une variable en utilisant l'algorithme RMSProp.

    Args:
        alpha: taux d'apprentissage
        beta2: facteur de décroissance pour la moyenne mobile
        epsilon: petit nombre pour éviter la division par zéro
        var: numpy.ndarray contenant la variable à mettre à jour
        grad: numpy.ndarray contenant le gradient de var
        s: moyenne mobile précédente du carré des gradients

    Returns:
        La variable mise à jour et la nouvelle moyenne mobile
    """
    # Mise à jour de la moyenne mobile du carré des gradients
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)
    
    # Calcul de la mise à jour de la variable
    var_new = var - alpha * grad / (np.sqrt(s_new) + epsilon)
    
    return var_new, s_new
