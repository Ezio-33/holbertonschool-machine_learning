#!/usr/bin/env python3
"""
Module pour créer un graphique à barres empilées montrant
la distribution des fruits par personne
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Crée un graphique à barres empilées représentant la quantité
    de différents fruits possédés par chaque personne.
    Les fruits sont empilés dans l'ordre: pommes, bananes, oranges, pêches.
    """
    # Génération des données
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    # Configuration de la figure
    plt.figure(figsize=(6.4, 4.8))

    # Définition des paramètres
    people = ['Farrah', 'Fred', 'Felicia']
    fruits = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    # Création des barres empilées
    bottom = np.zeros(3)
    for row, color, label in zip(fruit, colors, fruits):
        plt.bar(people, row, bottom=bottom, color=color,
                width=0.5, label=label)
        bottom += row

    # Configuration des axes et labels
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))

    # Ajout de la légende
    plt.legend()
    plt.show()
