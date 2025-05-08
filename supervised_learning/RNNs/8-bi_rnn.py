#!/usr/bin/env python3
"""
8-bi_rnn.py
Implémente la propagation avant complète d'un RNN bidirectionnel.
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Réalise la propagation avant d'un RNN bidirectionnel.

    Arguments
    ---------
    bi_cell : BidirectionalCell
        Instance déjà définie aux tasks 5-7.
    X : np.ndarray (t, m, i)
        Batch d'entrées.
    h_0 : np.ndarray (m, h)
        État caché initial (direction forward).
    h_t : np.ndarray (m, h)
        État caché initial (direction backward).

    Retours
    -------
    H : np.ndarray (t, m, 2*h)
        États cachés concaténés des deux directions.
    Y : np.ndarray (t, m, o)
        Sorties calculées par la cellule bidirectionnelle.
    """
    t, m, _ = X.shape
    _, h = h_0.shape

    # Direction FORWARD
    Hf = np.zeros((t, m, h))
    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        Hf[step] = h_prev

    #  Direction BACKWARD
    Hb = np.zeros((t, m, h))
    h_next = h_t
    for step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[step])
        Hb[step] = h_next

    # Concaténation
    H = np.concatenate((Hf, Hb), axis=2)

    #  Sorties
    Y = bi_cell.output(H)

    return H, Y
