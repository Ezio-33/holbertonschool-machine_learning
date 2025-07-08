#!/usr/bin/env python3
"""
This is the Temporal difference Project
it use the FrozenLake8x8-v1 environment
beware, do not slide on the hole
By Ced
"""
import numpy as np


def sample_episode(env, policy, max_steps=100):
    """
    Jouer un episode entier
    """

    SAR_list = []
    observation = 0  # le jouer debute en haut a gauche
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
    Utilise  Monte carlo, pour estimer la fonction de valeur
    """

    for episode in range(episodes):
        # reset the environment and sample one episode
        SAR_list = sample_episode(env, policy, max_steps)
        SAR_list = np.array(SAR_list, dtype=int)

        G = 0
        for state, reward in reversed(SAR_list):
            # return apres la fin de l'episode
            G = reward + gamma * G
            #print("G", G)
            # print("V", V[state])  # Commenté pour éviter trop de sorties
            # attention, si l'etat est nouveau ?!
            # Correction de l'erreur originale
            if episode == 0 or state not in SAR_list[:episode, 0]:
                # Update the value function V(s)
                V[state] = V[state] + alpha * (G - V[state])
    env.close()
    return V
