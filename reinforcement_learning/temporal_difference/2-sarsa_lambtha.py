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
    # Nombre d'états et d'actions
    n_states, n_actions = Q.shape

    def epsilon_greedy_policy(state, epsilon):
        """Politique epsilon-greedy pour choisir une action."""
        if np.random.random() < epsilon:
            return np.random.randint(n_actions)
        else:
            return np.argmax(Q[state])

    for episode in range(episodes):
        # Initialiser les traces d'éligibilité pour cet épisode
        eligibility_traces = np.zeros((n_states, n_actions))

        # Démarrer l'épisode
        state = 0
        env.reset()

        # Choisir l'action initiale avec la politique epsilon-greedy
        action = epsilon_greedy_policy(state, epsilon)

        for step in range(max_steps):
            # Exécuter l'action
            next_state, reward, done, truncated, _ = env.step(action)

            # Choisir la prochaine action avec la politique epsilon-greedy
            next_action = epsilon_greedy_policy(next_state, epsilon)

            # Calculer l'erreur TD
            td_error = (reward + gamma * Q[next_state, next_action] -
                        Q[state, action])

            # Mettre à jour les traces d'éligibilité
            eligibility_traces[state, action] += 1

            # Mettre à jour la table Q
            Q += alpha * td_error * eligibility_traces

            # Décrémenter les traces d'éligibilité
            eligibility_traces *= gamma * lambtha

            # Passer à l'état et action suivants
            state = next_state
            action = next_action

            if done or truncated:
                break

        # Mettre à jour epsilon après chaque épisode
        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    env.close()
    return Q
