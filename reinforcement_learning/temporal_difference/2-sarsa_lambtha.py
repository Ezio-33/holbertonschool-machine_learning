#!/usr/bin/env python3
"""
Module pour l'algorithme SARSA(λ)
"""
import numpy as np


def get_action(state, Q, epsilon):
    """
    Sélectionne une action en utilisant la politique epsilon-greedy.

    Args:
        state: État actuel
        Q: Table Q
        epsilon: Paramètre epsilon pour la politique epsilon-greedy
    Returns:
        action: Action sélectionnée
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, Q.shape[1])
    return np.argmax(Q[state, :])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Implémente l'algorithme SARSA(λ) pour l'apprentissage par renforcement.

    Args:
        env: Environnement
        Q: Table Q initiale
        lambtha: Paramètre λ pour les traces d'éligibilité
        episodes: Nombre d'épisodes
        max_steps: Nombre maximum d'étapes par épisode
        alpha: Taux d'apprentissage
        gamma: Facteur de discount
        epsilon: Valeur initiale d'epsilon pour la politique epsilon-greedy
        min_epsilon: Valeur minimale d'epsilon
        epsilon_decay: Taux de décroissance pour epsilon
    Returns:
        Q: Table Q mise à jour
    """
    if not (0 <= lambtha <= 1):
        raise ValueError("lambtha doit être entre 0 et 1")

    n_states, n_actions = Q.shape
    E = np.zeros((n_states, n_actions))
    initial_epsilon = epsilon

    for episode in range(episodes):
        E.fill(0)
        state = env.reset()[0]
        action = get_action(state, Q, epsilon)

        for step in range(max_steps):
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = get_action(next_state, Q, epsilon)

            delta = (reward + gamma * Q[next_state, next_action] -
                     Q[state, action])

            E[state, action] += 1

            Q += alpha * delta * E

            E *= gamma * lambtha

            state, action = next_state, next_action

            if done or truncated:
                break

        epsilon = max(min_epsilon,
                      min_epsilon + (initial_epsilon - min_epsilon) *
                      np.exp(-epsilon_decay * episode))

    return Q
