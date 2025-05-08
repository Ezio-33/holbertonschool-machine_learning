#!/usr/bin/env python3
"""7-bi_output.py
Cellule RNN bidirectionnelle : calcul des sorties à partir des états cachés.
"""

import numpy as np


class BidirectionalCell:
    """Représente une cellule RNN bidirectionnelle
    i : dimension des entrées
    h : dimension des états cachés
    o : dimension des sorties
    """

    def __init__(self, i: int, h: int, o: int):
        """Initialisation des poids et biais (loi normale N(0,1))"""
        # États forward et backward
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        # Sortie
        self.Wy = np.random.randn(2 * h, o)

        # Biais
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        """État caché forward d’un pas de temps"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        return np.tanh(concat @ self.Whf + self.bhf)

    def backward(self, h_next: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        """État caché backward d’un pas de temps (lecture inverse)"""
        concat = np.concatenate((h_next, x_t), axis=1)
        return np.tanh(concat @ self.Whb + self.bhb)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Soft-max numériquement stable sur le dernier axe"""
        x_shift = x - x.max(axis=-1, keepdims=True)
        exp = np.exp(x_shift)
        return exp / exp.sum(axis=-1, keepdims=True)

    def output(self, H: np.ndarray) -> np.ndarray:
        """Calcule toutes les sorties Y à partir des états concaténés
        Paramètres
        ----------
        H : np.ndarray (t, m, 2*h)
            États cachés concaténés forward + backward pour chaque pas.
        Retour
        ------
        Y : np.ndarray (t, m, o)
            Probabilités (soft-max) des sorties pour chaque pas de temps.
        """
        # Affine : logits (t, m, o)
        logits = np.matmul(H, self.Wy) + self.by
        # Soft-max
        return self._softmax(logits)
