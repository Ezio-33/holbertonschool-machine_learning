#!/usr/bin/env python3
"""Implémentation d'une cellule LSTM élémentaire (un pas de temps).

Classes
-------
LSTMCell
    Représente une cellule LSTM et effectue la propagation avant.
"""

import numpy as np


class LSTMCell:
    """Cellule LSTM pour réseaux récurrents.

    Paramètres
    ----------
    i : int
        Dimension de l'entrée `x_t`.
    h : int
        Dimension de l'état caché `h_t` et de l'état mémoire `c_t`.
    o : int
        Dimension de la sortie `y_t`.
    """

    def __init__(self, i: int, h: int, o: int):
        """Initialise poids et biais."""
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Fonction sigmoïde numérique stable."""
        return 1.0 / (1.0 + np.exp(-x))

    def forward(
        self,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
        x_t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Propagation avant d'un pas de temps.

        Paramètres
        ----------
        h_prev : ndarray, shape (m, h)
            État caché précédent.
        c_prev : ndarray, shape (m, h)
            État mémoire précédent.
        x_t : ndarray, shape (m, i)
            Entrée courante.

        Retour
        ------
        h_next : ndarray, shape (m, h)
            Nouvel état caché.
        c_next : ndarray, shape (m, h)
            Nouvel état mémoire.
        y : ndarray, shape (m, o)
            Sortie soft-max.
        """
        # 1) concaténation entrée + état caché
        concat = np.concatenate((h_prev, x_t), axis=1)

        # 2) calcul des portes
        f_t = self._sigmoid(concat @ self.Wf + self.bf)
        u_t = self._sigmoid(concat @ self.Wu + self.bu)
        c_bar = np.tanh(concat @ self.Wc + self.bc)
        o_t = self._sigmoid(concat @ self.Wo + self.bo)

        # 3) mise à jour de l'état mémoire
        c_next = f_t * c_prev + u_t * c_bar

        # 4) nouvel état caché
        h_next = o_t * np.tanh(c_next)

        # 5) sortie soft-max
        y_lin = h_next @ self.Wy + self.by
        exp = np.exp(y_lin - np.max(y_lin, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)

        return h_next, c_next, y
