#!/usr/bin/env python3
"""
Module pour l'algorithme SARSA(λ) (SARSA-Lambda)
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Implémente l'algorithme SARSA(λ) pour l'apprentissage par renforcement.

    Args:
        env: L'environnement dans lequel l'agent évolue.
        Q: Un tableau numpy de forme (s,a) contenant la table Q.
        lambtha: Le facteur de trace d'éligibilité.
        episodes: Le nombre total d'épisodes à simuler.
        max_steps: Le nombre maximum d'étapes par épisode.
        alpha: Le taux d'apprentissage.
        gamma: Le taux d'actualisation.
        epsilon: Le seuil initial pour epsilon greedy.
        min_epsilon: La valeur minimale vers laquelle epsilon doit décroître.
        epsilon_decay: Le taux de décroissance pour mettre à jour epsilon.

    Returns:
        Q: La table Q mise à jour.
    """
    n_states, n_actions = Q.shape
    initial_epsilon = epsilon
    E = np.zeros((n_states, n_actions))

    def get_action(state, Q, epsilon):
        """Choose action using epsilon-greedy policy"""
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(0, Q.shape[1])
        return np.argmax(Q[state, :])

    for episode in range(episodes):
        E.fill(0)
        state = env.reset()[0]
        action = get_action(state, Q, epsilon)
        steps = 0
        done = truncated = False

        while not (done or truncated) and steps <= max_steps:
            steps += 1

            next_state, reward, done, truncated, _ = env.step(action)

            next_action = get_action(next_state, Q, epsilon)

            delta = (reward + (gamma * Q[next_state, next_action]) -
                     Q[state, action])

            E[state, action] += 1
            Q += alpha * delta * E
            E *= gamma * lambtha

            state, action = next_state, next_action

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

    return Q
