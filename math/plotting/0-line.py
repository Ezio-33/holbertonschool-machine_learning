#!/usr/bin/env python3
"""
Module pour créer un graphique linéaire simple
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Fonction qui trace un graphique linéaire.
    La fonction trace y = x^3 pour x allant de 0 à 10
    avec une ligne rouge continue.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    x = np.arange(0, 11)
    plt.yticks([0, 200, 400, 600, 800, 1000])
    plt.xticks([0, 2, 4, 6, 8, 10])
    plt.plot(x, y, 'r-')
    plt.xlim(0, 10)
    plt.show()
