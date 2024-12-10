#!/usr/bin/env python3
"""
Module pour créer un nuage de points représentant
la relation entre la taille et le poids d'hommes
"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Crée un nuage de points montrant la relation entre
    la taille et le poids d'un échantillon d'hommes.

    Les données sont générées à partir d'une distribution
    normale multivariée.
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180

    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(x, y, color='magenta')
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")
    plt.show()
