#!/usr/bin/env python3
"""
4-deep_rnn.py
Implémente la propagation avant pour un RNN profond.

Pré-requis du projet :
- Python 3.9, NumPy 1.25.2
- Aucune importation externe (hors `import numpy as np`)
- Respect de pycodestyle 2.11.1
Toutes les chaînes de documentation et commentaires sont en français.
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Effectue la propagation avant d'un RNN profond.

    Paramètres
    ----------
    rnn_cells : list
        Liste des cellules RNN (longueur l).
    X : np.ndarray (t, m, i)
        Batch de séquences d'entrée.
    h_0 : np.ndarray (l, m, h)
        États cachés initiaux pour chaque couche.

    Retours
    -------
    H : np.ndarray (t+1, l, m, h)
        Tous les états cachés (H[0] = h_0).
    Y : np.ndarray (t, m, o)
        Toutes les sorties du dernier niveau.
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape

    # 1) tableau des états cachés
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    Y = None  # sera instancié après le premier pas de temps

    # 2) boucle temporelle
    for step in range(t):
        input_layer = X[step]

        # 2-a) boucle sur les couches
        for layer in range(l):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(
                h_prev, input_layer
            )
            H[step + 1, layer] = h_next
            input_layer = h_next

        # 2-b) empilage des sorties
        if Y is None:
            o = y.shape[1]
            Y = np.zeros((t, m, o))
        Y[step] = y

    return H, Y
