#!/usr/bin/env python3
"""
Module pour charger l'environnement FrozenLake.

Ce module contient une fonction pour charger l'environnement FrozenLake
avec une description personnalisée de la grille et un mode de rendu par défaut.
"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False,
                     render_mode="ansi"):
    """
    Charge l'environnement FrozenLake avec une description personnalisée
    et un mode de rendu par défaut.

    Args:
        desc (list of list of str): Description de la grille (par défaut None).
        map_name (str): Nom de la carte prédéfinie (par défaut None).
        is_slippery (bool): Si la glace est glissante (par défaut False).
        render_mode (str): Mode de rendu pour l'environnement
        (par défaut "ansi").

    Returns:
        gym.Env: L'environnement FrozenLake chargé.
    """
    return gym.make('FrozenLake-v1', desc=desc, map_name=map_name,
                    is_slippery=is_slippery, render_mode=render_mode)
