#!/usr/bin/env python3
"""
Implémentation d'une cellule RNN simple.
Toutes les chaînes de documentation sont rédigées en français
conformément aux exigences du projet.
"""
import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Calcule la fonction softmax colonne par colonne.

    Args:
        z (np.ndarray): scores linéaires de forme (m, o)

    Returns:
        np.ndarray: probabilités normalisées de même forme
    """
    exp_z = np.exp(z - z.max(axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)


class RNNCell:
    """
    Représente une cellule d'un RNN simple.
    """

    def __init__(self, i: int, h: int, o: int):
        """
        Initialise la cellule.

        Args:
            i (int): dimension des données d'entrée
            h (int): dimension de l'état caché
            o (int): dimension de la sortie
        """
        # Poids pour l'état caché : [h_prev ; x_t] -> h_next
        self.Wh = np.random.randn(i + h, h)

        # Poids pour la sortie : h_next -> y
        self.Wy = np.random.randn(h, o)

        # Biais correspondants
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev: np.ndarray, x_t: np.ndarray):
        """
        Effectue la propagation avant pour un pas de temps.

        Args:
            h_prev (np.ndarray): état caché précédent, forme (m, h)
            x_t   (np.ndarray): donnée du pas de temps, forme (m, i)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - h_next (np.ndarray): nouvel état caché, forme (m, h)
                - y      (np.ndarray): sortie, forme (m, o)
        """
        # Concaténation des entrées
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Calcul de l'état caché suivant
        h_next = np.tanh(concat @ self.Wh + self.bh)

        # Calcul de la sortie non normalisée
        y_lin = h_next @ self.Wy + self.by

        # Application du softmax pour obtenir des probabilités
        y = softmax(y_lin)

        # 5. Renvoi du résultat
        return h_next, y
