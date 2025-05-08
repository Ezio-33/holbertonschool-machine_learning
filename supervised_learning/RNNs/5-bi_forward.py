#!/usr/bin/env python3
"""
BidirectionalCell : cellule RNN bidirectionnelle — sens forward uniquement.
"""

import numpy as np


class BidirectionalCell:
    """
    Implémente la partie forward d’un RNN bidirectionnel.

    Paramètres
    ----------
    i : int
        Dimension des données d’entrée x_t.
    h : int
        Dimension des états cachés.
    o : int
        Dimension des sorties y_t (sera utile dans les tâches suivantes).
    """

    def __init__(self, i: int, h: int, o: int):
        """Initialise les poids et biais."""
        # Poids pour la direction forward : concat(h_prev, x_t) → h
        self.Whf = np.random.randn(i + h, h)
        # Poids pour la direction backward
        self.Whb = np.random.randn(i + h, h)
        # Poids pour la sortie
        self.Wy = np.random.randn(2 * h, o)

        # Biais correspondants
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        """
        Calcule l’état caché forward pour un pas de temps.

        Paramètres
        ----------
        h_prev : ndarray (m, h)
            État caché précédent.
        x_t : ndarray (m, i)
            Entrée courante.

        Retour
        ------
        h_next : ndarray (m, h)
            Nouvel état caché forward.
        """
        # 1. Concaténation le long des features
        concat = np.concatenate((h_prev, x_t), axis=1)

        # 2. Transformation linéaire + biais
        lin = np.matmul(concat, self.Whf) + self.bhf

        # 3. Activation tanh
        h_next = np.tanh(lin)

        return h_next
