#!/usr/bin/env python3
"""
Module pour initialiser la table Q pour l'environnement FrozenLake.

Ce module contient une fonction pour initialiser la table Q avec des zéros,
en fonction du nombre d'états et d'actions dans l'environnement FrozenLake.
"""

import numpy as np


def q_init(env):
    """
    Initialise la table Q pour un environnement donné.

    Args:
        env (gym.Env): L'environnement FrozenLake pour lequel
        initialiser la Q-table.

    Returns:
        numpy.ndarray: La Q-table initialisée avec des zéros.
            La Q-table est une matrice 2D où le nombre de lignes est égal au
            nombre d'états dans l'environnement, et le nombre de colonnes est
            égal au nombre d'actions possibles.
            Chaque entrée Q[s, a] représente la valeur estimée de prendre
            l'action 'a' dans l'état 's'.
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    Q = np.zeros((num_states, num_actions))

    return Q
