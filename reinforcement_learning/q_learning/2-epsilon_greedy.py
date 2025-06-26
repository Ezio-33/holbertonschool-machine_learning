#!/usr/bin/env python3
"""
Module pour implémenter la stratégie epsilon-greedy
dans le contexte de Q-learning.

Ce module contient une fonction pour choisir une action
en utilisant la stratégie epsilon-greedy, qui équilibre
exploration et exploitation.
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Utilise la stratégie epsilon-greedy pour déterminer la prochaine action.

    Args:
        Q (numpy.ndarray): La Q-table contenant les valeurs
                Q pour chaque état-action.
        state (int): L'état actuel pour lequel choisir une action.
        epsilon (float): Valeur entre 0 et 1 qui contrôle
                le compromis exploration/exploitation.

    Returns:
        int: L'indice de l'action choisie.

    Description:
        Avec une probabilité epsilon, une action aléatoire
                est choisie (exploration).
        Avec une probabilité (1 - epsilon), l'action avec la valeur
                Q la plus élevée pour l'état actuel est choisie (exploitation).
    """
    state_Q_values = Q[state]

    random_value = np.random.uniform(0, 1)

    if random_value < epsilon:
        action = np.random.randint(0, len(state_Q_values))
    else:
        action = np.argmax(state_Q_values)

    return action
