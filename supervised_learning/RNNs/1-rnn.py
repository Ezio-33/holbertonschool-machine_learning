#!/usr/bin/env python3
"""
Module 1-rnn
Implémente la propagation avant d’un RNN simple.

Pré-requis :
    * Python 3.9
    * NumPy 1.25.2
    * Style pycodestyle 2.11.1
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Effectue la propagation avant sur toute la séquence.

    Args:
        rnn_cell (RNNCell): instance déjà entraînée ou initialisée.
        X (ndarray): données d’entrée de forme (t, m, i).
        h_0 (ndarray): état caché initial de forme (m, h).

    Returns:
        H (ndarray): tous les états cachés, forme (t + 1, m, h)
                     où H[0] == h_0.
        Y (ndarray): toutes les sorties,   forme (t, m, o)
    """
    t, m, _ = X.shape
    _, h = h_0.shape

    # Listes temporaires pour accumuler résultats
    H = np.zeros((t + 1, m, h))
    H[0] = h_0
    Y_list = []

    h_prev = h_0
    for step in range(t):
        x_t = X[step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next
        Y_list.append(y)
        h_prev = h_next

    Y = np.stack(Y_list, axis=0)
    return H, Y
