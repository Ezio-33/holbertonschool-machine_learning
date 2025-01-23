#!/usr/bin/env python3
"""
Module contenant la fonction d'optimisation Adam
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Met à jour une variable en utilisant l'algorithme d'optimisation Adam.

    Args:
        alpha: taux d'apprentissage
        beta1: facteur pour la moyenne mobile du premier moment
        beta2: facteur pour la moyenne mobile du second moment
        epsilon: petit nombre pour éviter la division par zéro
        var: numpy.ndarray contenant la variable à mettre à jour
        grad: numpy.ndarray contenant le gradient de var
        v: premier moment précédent
        s: second moment précédent
        t: pas de temps pour la correction du biais

    Returns:
        var_new: variable mise à jour
        v_new: nouveau premier moment
        s_new: nouveau second moment
    """
    # Calcul des moments
    v_new = beta1 * v + (1 - beta1) * grad
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    # Correction du biais
    v_corrected = v_new / (1 - beta1 ** t)
    s_corrected = s_new / (1 - beta2 ** t)

    # Mise à jour de la variable
    var_new = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var_new, v_new, s_new
