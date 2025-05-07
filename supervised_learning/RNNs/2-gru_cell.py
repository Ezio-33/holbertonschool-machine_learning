#!/usr/bin/env python3
"""
Implémente une cellule GRU (Gated Recurrent Unit) en NumPy.

Cette classe sert à un unique pas de temps :
    - entrée x_t  : (m, i)
    - état h_prev : (m, h)
Elle renvoie le nouvel état h_next et la sortie y (soft-max).
"""

import numpy as np


def _sigmoid(x):
    """Sigmoïde (fonction logistique) appliquée élément-par-élément."""
    return 1 / (1 + np.exp(-x))


def _softmax(x):
    """Soft-max stable numériquement, appliqué ligne par ligne."""
    x_shift = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x_shift)
    return exp / exp.sum(axis=1, keepdims=True)


class GRUCell:
    """
    Cellule GRU minimale.

    Attributs publics créés :
        Wz, Wr, Wh, Wy : matrices de poids
        bz, br, bh, by : biais associés
    """

    def __init__(self, i, h, o):
        """
        Initialise tous les poids et biais.

        Args:
            i (int): dimension de l'entrée x_t
            h (int): dimension de l'état caché h_prev / h_next
            o (int): dimension de la sortie y
        """
        # --- Poids
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # --- Biais
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calcule h_next et y pour un pas de temps.

        Args:
            h_prev (np.ndarray): état caché précédent, shape (m, h)
            x_t    (np.ndarray): entrée courante,     shape (m, i)

        Returns:
            tuple: (h_next, y)
                h_next (np.ndarray): nouvel état caché, shape (m, h)
                y      (np.ndarray): sortie soft-max,  shape (m, o)
        """
        # Concaténation h_prev | x_t ➜ (m, h+i)
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Portes update (z) et reset (r)
        z = _sigmoid(concat @ self.Wz + self.bz)
        r = _sigmoid(concat @ self.Wr + self.br)

        # État candidat h̃
        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(concat_r @ self.Wh + self.bh)

        # Nouvel état caché
        h_next = (1 - z) * h_prev + z * h_tilde

        # Sortie soft-max
        y_logits = h_next @ self.Wy + self.by
        y = _softmax(y_logits)

        return h_next, y
