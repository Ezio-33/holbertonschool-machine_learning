#!/usr/bin/env python3
"""
Script pour tester un agent avec des poids aléatoires (comparaison)
"""
import os
import warnings

# Suppression des warnings TensorFlow
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

    print("=== TEST AVEC AGENT ALÉATOIRE (COMPARAISON) ===")
    print("Cet agent n'a pas été entraîné - comportement de référence")
    
    dqn.test(env, nb_episodes=3, visualize=True, verbose=2)
    
    print("=== TEST TERMINÉ ===")
    env.close()
