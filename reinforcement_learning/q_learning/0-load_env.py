#!/usr/bin/env python3
"""
Module pour charger l'environnement FrozenLake de gymnasium.
"""

import gymnasium as gym

def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Charge l'environnement FrozenLakeEnv depuis gymnasium.

    Args:
        desc (list of lists, optional): Description personnalisée de la carte.
        map_name (str, optional): Nom de la carte pré-faite à charger.
        is_slippery (bool, optional): Détermine si la glace est glissante.

    Returns:
        gym.Env: L'environnement FrozenLake chargé.
    """
    env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name,
                   is_slippery=is_slippery)
    return env
