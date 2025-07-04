#!/usr/bin/env python3
"""
Module d'entraînement d'un agent Deep Q-Learning pour le jeu Breakout d'Atari

Ce script implémente un agent d'apprentissage par renforcement
utilisant l'algorithme Deep Q-Network (DQN) pour apprendre à jouer
au jeu Breakout. L'agent utilise un réseau de neurones convolutif
pour traiter les images du jeu et prendre des décisions optimales.

Auteur: Samuel VERSCHUEREN
Date: 03 Juillet 2025
Version: 1.0
"""
from PIL import Image
import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor



class GymWrapper:
    """
    Wrapper de compatibilité pour l'environnement Gym

    Cette classe assure la compatibilité entre les nouvelles versions de Gym
    (qui retournent des tuples) et la bibliothèque keras-rl2 qui s'attend à
    recevoir directement les observations.

    Attributes:
        env: L'environnement Gym original
        action_space: L'espace des actions possibles
        observation_space: L'espace des observations
    """

    def __init__(self, env):
        """
        Initialise le wrapper avec l'environnement Gym

        Args:
            env: L'environnement Gym à wrapper
        """
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        """
        Remet l'environnement à zéro et retourne l'observation initiale

        Returns:
            observation: L'observation initiale du jeu (sans les métadonnées)
        """
        obs, info = self.env.reset()
        return obs

    def step(self, action):
        """
        Exécute une action dans l'environnement

        Args:
            action: L'action à exécuter

        Returns:
            tuple: (observation, reward, done, info) où done combine
                   les conditions de fin de partie et de troncature
        """
        obs, reward, done, truncated, info = self.env.step(action)
        # Combine done et truncated pour la compatibilité avec keras-rl2
        return obs, reward, done or truncated, info

    def render(self, mode='human'):
        """
        Affiche l'état actuel du jeu

        Args:
            mode: Mode de rendu (défaut: 'human')
        """
        return self.env.render()

    def close(self):
        """Ferme l'environnement et libère les ressources"""
        return self.env.close()


class AtariProcessor(Processor):
    """
    Préprocesseur spécialisé pour les jeux Atari

    Cette classe traite les observations brutes du jeu Atari pour les rendre
    compatibles avec l'entraînement du réseau de neurones. Elle effectue le
    redimensionnement, la conversion en niveaux de gris et la normalisation.
    """

    def process_observation(self, observation):
        """
        Prétraite une observation individuelle

        Convertit l'image couleur 210x160x3 en image en niveaux de gris 84x84
        pour réduire la complexité computationnelle tout en conservant
        l'information essentielle pour le jeu.

        Args:
            observation: Image brute du jeu (numpy array de forme
            (210, 160, 3))

        Returns:
            processed_observation: Image prétraitée
            (numpy array de forme (84, 84))
        """
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        # Redimensionnement à 84x84 et conversion en niveaux de gris
        img = img.resize((84, 84)).convert('L')
        # Conversion en array numpy
        processed_observation = np.array(img)
        assert processed_observation.shape == (84, 84)
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """
        Normalise un lot d'états pour l'entraînement

        Convertit les valeurs de pixels de l'intervalle [0, 255] vers [0, 1]
        pour améliorer la stabilité et la convergence de l'entraînement.

        Args:
            batch: Lot d'états (images) à normaliser

        Returns:
            processed_batch: Lot d'états normalisés avec valeurs entre 0 et 1
        """
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """
        Limite les récompenses pour stabiliser l'apprentissage

        Applique un clipping des récompenses dans l'intervalle [-1, 1]
        pour éviter les gradients explosifs et améliorer la stabilité
        de l'algorithme DQN.

        Args:
            reward: Récompense brute de l'environnement

        Returns:
            reward: Récompense limitée entre -1 et 1
        """
        return np.clip(reward, -1., 1.)


def build_model(num_action):
    """
    Construit l'architecture du réseau de neurones convolutif pour le DQN

    Cette fonction crée un CNN. Le réseau prend en entrée
    4 images consécutives de 84x84 pixels et produit les Q-values
    pour chaque action possible.

    Architecture:
    - Couche de permutation pour réorganiser les dimensions
    - 3 couches convolutives avec activation ReLU
    - 2 couches entièrement connectées
    - Couche de sortie linéaire pour les Q-values

    Args:
        num_action: Nombre d'actions possibles dans l'environnement

    Returns:
        model: Modèle Keras compilé prêt pour l'entraînement
    """
    # Format d'entrée: 4 images de 84x84 pixels
    input_shape = (4, 84, 84)
    model = Sequential()

    # Réorganisation des dimensions pour TensorFlow
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(num_action))
    model.add(Activation('linear'))
    return model


if __name__ == '__main__':

    env = gym.make("Breakout-v0")
    env = GymWrapper(env)
    observation = env.reset()
    num_action = env.action_space.n
    window = 4

    model = build_model(num_action)
    model.summary()

    # === CONFIGURATION DE LA MÉMOIRE ET DU PRÉPROCESSEUR ===

    memory = SequentialMemory(limit=1000000, window_length=4)
    # Préprocesseur pour les images Atari
    processor = AtariProcessor()

    # === CONFIGURATION DE LA POLITIQUE D'EXPLORATION ===
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=0.05, value_test=0.01,
                                  nb_steps=50000)

    # === CRÉATION ET CONFIGURATION DE L'AGENT DQN ===
    dqn = DQNAgent(model=model, nb_actions=num_action, policy=policy,
                   memory=memory, processor=processor,
                   nb_steps_warmup=10000,
                   gamma=.99,
                   target_model_update=1000,
                   train_interval=4,
                   delta_clip=1.)

    dqn.compile(Adam(learning_rate=.00025), metrics=['mae'])

    # === ENTRAÎNEMENT DE L'AGENT ===
    print("=== DÉBUT DE L'ENTRAÎNEMENT ===")
    print("Cela peut prendre plusieurs heures...")
    
    dqn.fit(env,
            nb_steps=1000000,
            log_interval=50000,
            visualize=False,
            verbose=2,
            nb_max_episode_steps=10000)
    
    dqn.save_weights('policy.h5', overwrite=True)
