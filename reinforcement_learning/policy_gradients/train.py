#!/usr/bin/env python3
"""
Ce projet concerne le gradient de politique
Par Ced
"""
import numpy as np
import gymnasium as gym
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Tâche 3, entraînement de l'agent cartpole
    affiche le score à chaque épisode
    affiche en mode humain tous les 1000 épisodes
    """

    # tableau de scores
    scores = []

    # obtenir le nombre d'états et d'actions de l'environnement gymnasium
    n_states, n_actions = env.observation_space.shape[0], env.action_space.n

    weights = np.random.rand(n_states, n_actions)

    for i in range(nb_episodes):

        # définir done à False pour l'état initial
        done = False

        # afficher un épisode tous les 1000 épisodes
        # si show_result est défini à True
        if show_result and i % 1000 == 0:
            env = gym.make('CartPole-v1', render_mode="human")
        else:
            env = gym.make('CartPole-v1', render_mode=None)

        state, _ = env.reset()
        rewards = []
        gradients = []

        while not done:
            # obtenir l'action et le gradient
            action, grad = policy_gradient(state, weights)
            state, reward, done, truncated, _ = env.step(action)

            # appliquer le gradient de politique
            gradients.append(grad)
            rewards.append(reward)

            weights += alpha * sum([g * (gamma ** t) * r for t, (g, r) in enumerate(zip(gradients, rewards))])

            # quitter après 500 étapes
            if done:
                break

        # fermer chaque épisode, fermer la fenêtre
        env.close()
        scores.append(sum(rewards))

        # afficher le score à chaque épisode
        print("EP: " + str(i) + " Score: " + str(sum(rewards)))

    return scores