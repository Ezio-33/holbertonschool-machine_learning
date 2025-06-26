#!/usr/bin/env python3
"""
Module pour entraîner un agent à jouer à FrozenLake en utilisant Q-learning.

Ce module contient une fonction pour entraîner
un agent en utilisant l'algorithme Q-learning avec une stratégie
epsilon-greedy pour l'exploration/exploitation.
"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Entraîne un agent à jouer à FrozenLake en utilisant Q-learning.

    Args:
        env: L'environnement FrozenLake.
        Q (numpy.ndarray): La Q-table initiale.
        episodes (int): Nombre total d'épisodes d'entraînement.
        max_steps (int): Nombre maximum d'étapes par épisode.
        alpha (float): Taux d'apprentissage.
        gamma (float): Facteur d'escompte pour les récompenses futures.
        epsilon (float): Valeur initiale pour epsilon-greedy.
        min_epsilon (float): Valeur minimale pour epsilon.
        epsilon_decay (float): Taux de décroissance
                exponentielle pour epsilon.

    Returns:
        tuple: (Q, total_rewards)
            Q (numpy.ndarray): La Q-table entraînée.
            total_rewards (list): Liste des récompenses
                        totales obtenues à chaque épisode.
    """
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done and reward == 0:
                reward = -1

            Q[state][action] = Q[state][action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state][action]
            )

            state = new_state
            episode_reward += reward

            if done:
                break

        epsilon = min_epsilon + (epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode
        )

        total_rewards.append(episode_reward)

    return Q, total_rewards
