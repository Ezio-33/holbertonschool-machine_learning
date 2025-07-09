#!/usr/bin/env python3
"""
Module pour l'algorithme TD(λ) (TD-Lambda)
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Implémente l'algorithme TD(λ) pour estimer la fonction de valeur.

    Args:
        env: L'environnement dans lequel l'agent évolue.
        V: Un tableau numpy contenant les estimations de valeur.
        policy: La politique à suivre pour choisir les actions.
        lambtha: Le facteur de trace d'éligibilité.
        episodes: Le nombre total d'épisodes à simuler.
        max_steps: Le nombre maximum d'étapes par épisode.
        alpha: Le taux d'apprentissage.
        gamma: Le taux d'actualisation.

    Returns:
        V: Le tableau des estimations de valeur mis à jour.
    """
    # Nombre d'états dans l'environnement
    n_states = len(V)

    for episode in range(episodes):
        # Initialiser les traces d'éligibilité pour cet épisode
        eligibility_traces = np.zeros(n_states)

        # Démarrer l'épisode
        state = 0
        env.reset()

        for step in range(max_steps):
            action = policy(state)

            next_state, reward, done, truncated, _ = env.step(action)

            td_error = reward + gamma * V[next_state] - V[state]

            eligibility_traces[state] += 1

            V += alpha * td_error * eligibility_traces

            eligibility_traces *= gamma * lambtha

            state = next_state

            if done or truncated:
                break

    env.close()
    return V
