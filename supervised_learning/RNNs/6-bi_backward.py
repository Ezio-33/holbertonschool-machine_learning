#!/usr/bin/env python3
"""
BidirectionalCell : cellule RNN bidirectionnelle
Implémente la propagation avant (forward) et arrière (backward)
dans le cadre du projet Holberton « RNNs ».
"""

import numpy as np


class BidirectionalCell:
    """
    Représente une cellule RNN bidirectionnelle.

    Attributs publics
    -----------------
    Whf, bhf : poids/biais direction avant
    Whb, bhb : poids/biais direction arrière
    Wy,  by  : poids/biais pour la couche de sortie
    """

    def __init__(self, i: int, h: int, o: int):
        """
        Paramètres
        ----------
        i : int
            Dimension des vecteurs d'entrée x_t.
        h : int
            Dimension des états cachés h_t.
        o : int
            Dimension des sorties y_t.
        """
        # Poids pour la direction avant : concat([h_prev, x_t]) → h_next
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))

        # Poids pour la direction arrière : concat([h_next, x_t]) → h_prev
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))

        # Poids pour la sortie : concat([h_fwd, h_bwd]) → y_t
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        """
        Calcule l'état caché h_next pour un pas (sens avant).

        h_prev : ndarray (m, h) – état caché précédent
        x_t    : ndarray (m, i) – entrée courante
        Retour : ndarray (m, h) – h_next
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        lin = np.matmul(concat, self.Whf) + self.bhf
        h_next = np.tanh(lin)
        return h_next

    def backward(self, h_next: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        """
        Calcule l'état caché h_prev pour un pas (sens arrière).

        h_next : ndarray (m, h) – état caché "suivant" (t+1 en marche arrière)
        x_t    : ndarray (m, i) – entrée courante
        Retour  : ndarray (m, h) – h_prev (t-1 en marche arrière)
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        lin = np.matmul(concat, self.Whb) + self.bhb
        h_prev = np.tanh(lin)
        return h_prev
