#!/usr/bin/env python3
"""
Module contenant la fonction de mise à jour avec momentum
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Met à jour une variable en utilisant l'algorithme de
    gradient descent avec momentum.

    Args:
        alpha: taux d'apprentissage (learning rate)
        beta1: facteur de momentum
        var: numpy.ndarray contenant la variable à mettre à jour
        grad: numpy.ndarray contenant le gradient de var
        v: moment précédent de var

    Returns:
        La variable mise à jour et le nouveau moment
    """
    # Calcul du nouveau moment
    v_new = beta1 * v + (1 - beta1) * grad

    # Mise à jour de la variable
    var_new = var - alpha * v_new

    return var_new, v_new
