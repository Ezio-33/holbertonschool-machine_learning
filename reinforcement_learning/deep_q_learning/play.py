#!/usr/bin/env python3
"""
Script pour utiliser un agent préalablement entraîné
capable de jouer à Breakout d'Atari.
"""
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from tensorflow.keras.optimizers.legacy import Adam
from train import build_model, AtariProcessor, GymWrapper

if __name__ == '__main__':
    env = gym.make("Breakout-v0", render_mode="human")
    env = GymWrapper(env)
    observation = env.reset()

    num_actions = env.action_space.n
    model = build_model(num_actions)

    memory = SequentialMemory(limit=1000000, window_length=4)
    processor = AtariProcessor()

    dqn = DQNAgent(
        model=model,
        nb_actions=num_actions,
        policy=GreedyQPolicy(),
        processor=processor,
        memory=memory
    )

    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    dqn.load_weights('policy.h5')

    print("=== DÉBUT DU TEST DE L'AGENT ENTRAÎNÉ ===")
    print("Note: Si l'agent semble aléatoire, l'entraînement était insuffisant")
    
    dqn.test(env, nb_episodes=5, visualize=True, verbose=2)
    
    print("=== TEST TERMINÉ ===")
    env.close()
