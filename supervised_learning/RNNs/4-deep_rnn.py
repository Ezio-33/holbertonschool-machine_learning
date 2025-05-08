#!/usr/bin/env python3
"""Propagation avant d’un RNN profond (plusieurs couches)."""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Exécute la propagation avant d’un RNN profond.

    Paramètres
    ----------
    rnn_cells : list
        Liste ordonnée des cellules RNN (longueur = n_layers).
    X : ndarray shape (t_steps, m, i)
        Séquences d’entrée :
          t_steps = nombre de pas de temps,
          m       = taille du batch,
          i       = dimension d’un vecteur d’entrée.
    h_0 : ndarray shape (n_layers, m, h_dim)
        États cachés initiaux pour chaque couche
        (h_dim = dimension d’un état caché).

    Retours
    -------
    H : ndarray shape (t_steps + 1, n_layers, m, h_dim)
        Tous les états cachés (H[0] == h_0).
    Y : ndarray shape (t_steps, m, o_dim)
        Sorties de la **dernière** couche à chaque pas de temps
        (o_dim = dimension de sortie d’une cellule).
    """
    # -------- dimensions utiles ---------------------------------------------
    t_steps, m, _ = X.shape
    n_layers, _, h_dim = h_0.shape

    # -------- mémoires -------------------------------------------------------
    H = np.zeros((t_steps + 1, n_layers, m, h_dim))
    H[0] = h_0

    _, o_dim = rnn_cells[-1].Wy.shape
    Y = np.zeros((t_steps, m, o_dim))

    for layer_idx, cell in enumerate(rnn_cells):

        for t in range(1, t_steps + 1):
            # -------- entrée de la couche courante --------------------------
            if layer_idx == 0:
                x_t = X[t - 1]
            else:
                x_t = H[t, layer_idx - 1]

            # -------- état caché précédent de cette couche ------------------
            h_prev = H[t - 1, layer_idx]

            # -------- appel de la cellule -----------------------------------
            h_next, y = cell.forward(h_prev, x_t)

            H[t, layer_idx] = h_next

            # -------- stocker la sortie finale ------------------------------
            if layer_idx == n_layers - 1:
                Y[t - 1] = y

    return H, Y
