#!/usr/bin/env python3
"""
Module pour jouer à FrozenLake en utilisant une Q-table entraînée.

Ce module contient une fonction pour utiliser une Q-table entraînée pour jouer
à FrozenLake de manière optimale, en choisissant toujours l'action
avec la valeur Q la plus élevée.
"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Joue une partie complète de FrozenLake en utilisant une Q-table entraînée.

    Args:
        env: L'environnement FrozenLake.
        Q (numpy.ndarray): La Q-table entraînée.
        max_steps (int): Nombre maximum d'étapes dans le jeu (par défaut 100).

    Returns:
        tuple: (reward, render)
            reward (float): La récompense obtenue à la fin du jeu.
            render (list): Liste des représentations du jeu à chaque étape.
    """
    state, _ = env.reset()
    render = []

    for step in range(max_steps):
        graph = env.render()
        render.append(graph)
        action = np.argmax(Q[state])
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = new_state
        if done:
            break

    graph = env.render()
    render.append(graph)
    env.close()

    return reward, render
