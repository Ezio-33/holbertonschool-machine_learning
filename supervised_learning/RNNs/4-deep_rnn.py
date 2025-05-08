#!/usr/bin/env python3
"""
4-deep_rnn.py
Propagation avant d’un RNN profond (plusieurs couches).

Hypothèses :
- Python 3.9  •  NumPy 1.25.2
- Seul import autorisé : « import numpy as np »
- Respect de pycodestyle 2.11.1
- Docstrings et commentaires en français.
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Exécute la propagation avant d’un RNN profond.

    Paramètres
    ----------
    rnn_cells : list
        Liste ordonnée des cellules RNN (longueur l).
    X : ndarray shape (t, m, i)
        Séquences d’entrée :
          t = nombre de pas de temps,
          m = taille du batch,
          i = dimension d’un vecteur d’entrée.
    h_0 : ndarray shape (l, m, h)
        États cachés initiaux de chaque couche
        (h = dimension d’un état caché).

    Retours
    -------
    H : ndarray shape (t + 1, l, m, h)
        États cachés pour tous les pas et toutes les couches
        (H[0] == h_0).
    Y : ndarray shape (t, m, o)
        Sorties du dernier niveau pour chaque pas de temps
        (o = dimension de sortie d’une cellule).
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape

    # 1) mémoire pour les états cachés
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    Y = None

    # 2) boucle temporelle
    for step in range(t):
        layer_input = X[step]

        # 2-a) boucle sur les couches
        for layer in range(l):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, layer_input)

            H[step + 1, layer] = h_next
            layer_input = h_next

        # 2-b) empilage des sorties (une seule fois pour initialiser Y)
        if Y is None:
            o = y.shape[1]
            Y = np.zeros((t, m, o))
        Y[step] = y
    return H, Y
