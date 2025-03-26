#!/usr/bin/env python3
"""Module définissant la classe MultiNormal pour une
distribution normale multivariée."""

import numpy as np


class MultiNormal:
    """
    Classe représentant une distribution normale multivariée.
    """

    def __init__(self, data):
        """
        Initialise une instance de MultiNormal.

        Args:
            data (numpy.ndarray): Tableau de forme (d, n)
            contenant les données.
            d est le nombre de dimensions, n est
            le nombre de points de données.

        Raises:
            TypeError: Si data n'est pas un numpy.ndarray 2D.
            ValueError: Si data contient moins de 2 points de données.
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        centered_data = data - self.mean
        self.cov = np.dot(centered_data, centered_data.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """
        Calcule la fonction de densité de probabilité (PDF) en un point donné.

        Args:
            x (numpy.ndarray): Point de données de forme (d, 1).

        Returns:
            float: Valeur de la PDF au point x.

        Raises:
            TypeError: Si x n'est pas un numpy.ndarray.
            ValueError: Si x n'a pas la bonne forme.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.cov.shape[0]
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        x_m = x - self.mean
        cov_inv = np.linalg.inv(self.cov)
        exponent = -0.5 * np.dot(np.dot(x_m.T, cov_inv), x_m)
        coefficient = 1 / ((2 * np.pi) ** (d / 2) *
                           np.sqrt(np.linalg.det(self.cov)))

        return float(coefficient * np.exp(exponent))
