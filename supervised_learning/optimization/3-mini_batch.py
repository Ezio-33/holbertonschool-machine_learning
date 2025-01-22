#!/usr/bin/env python3
"""
Module contenant la fonction de création de mini-batches
pour l'entraînement d'un réseau de neurones
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Crée des mini-batches à partir des données d'entrée.

    Args:
        X: numpy.ndarray de forme (m, nx) contenant les données d'entrée
            m est le nombre de données
            nx est le nombre de caractéristiques
        Y: numpy.ndarray de forme (m, ny) contenant les étiquettes
            m est le même nombre de données que X
            ny est le nombre de classes
        batch_size: nombre de données dans chaque batch

    Returns:
        Une liste de tuples (X_batch, Y_batch)
    """
    m = X.shape[0]
    
    # Mélanger les données
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    
    # Créer les mini-batches
    mini_batches = []
    n_complete_batches = m // batch_size
    
    # Créer les batches complets
    for i in range(n_complete_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        mini_batch_X = X_shuffled[start_idx:end_idx]
        mini_batch_Y = Y_shuffled[start_idx:end_idx]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    
    # Gérer le dernier batch s'il est incomplet
    if m % batch_size != 0:
        start_idx = n_complete_batches * batch_size
        mini_batch_X = X_shuffled[start_idx:]
        mini_batch_Y = Y_shuffled[start_idx:]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    
    return mini_batches
