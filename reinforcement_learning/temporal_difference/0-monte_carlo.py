#!/usr/bin/env python3
"""
Module pour l'algorithme Monte Carlo
"""
import numpy as np


def sample_episode(env, policy, max_steps=100):
    """
    Simule un épisode complet en suivant la politique donnée.

    Args:
        env: L'environnement dans lequel l'agent évolue.
        policy: La politique à suivre pour choisir les actions.
        max_steps: Le nombre maximum d'étapes par épisode.

    Returns:
        SAR_list: Une liste de tuples (état, récompense).
    """
    SAR_list = []
    observation = 0
    env.reset()
    for j in range(max_steps):
        action = policy(observation)
        new_obs, reward, done, truncated, _ = env.step(action)
        SAR_list.append((observation, reward))

        if done or truncated:
            break

        observation = new_obs
    return SAR_list


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Implémente l'algorithme Monte Carlo pour estimer la fonction de valeur.

    Args:
        env: L'environnement dans lequel l'agent évolue.
        V: Un tableau numpy contenant les estimations de valeur.
        policy: La politique à suivre pour choisir les actions.
        episodes: Le nombre total d'épisodes à simuler.
        max_steps: Le nombre maximum d'étapes par épisode.
        alpha: Le taux d'apprentissage.
        gamma: Le taux d'actualisation.

    Returns:
        V: Le tableau des estimations de valeur mis à jour.
    """

    for episode in range(episodes):
        SAR_list = sample_episode(env, policy, max_steps)
        SAR_list = np.array(SAR_list, dtype=int)

        G = 0
        for state, reward in reversed(SAR_list):
            G = reward + gamma * G
            if state not in SAR_list[:episode, 0]:
                V[state] = V[state] + alpha * (G - V[state])
    env.close()
    return V
