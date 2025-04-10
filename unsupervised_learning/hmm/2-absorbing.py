#!/usr/bin/env python3
"""
Module pour déterminer si une chaîne de Markov est absorbante
Contient la fonction absorbing
"""
import numpy as np


def absorbing(P):
    """
    Détermine si une chaîne de Markov est absorbante

    Args:
        P (np.ndarray): Matrice de transition (n x n)

    Returns:
        bool: True si absorbante, False sinon
    """
    # Vérification de la forme de la matrice
    if not isinstance(
            P,
            np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False

    n = P.shape[0]  # Nombre d'états

    # Cas spécial pour matrice vide
    if n == 0:
        return False

    # Identification des états absorbants
    absorbants = []
    for i in range(n):
        # Vérifie si l'élément diagonal est ~1 et les autres ~0
        if np.isclose(P[i, i], 1.0):
            ligne_sans_diag = np.delete(P[i], i)
            if np.allclose(ligne_sans_diag, 0):
                absorbants.append(i)

    # Aucun état absorbant trouvé
    if not absorbants:
        return False

    # Construction de la liste d'adjacence (transitions possibles)
    graphe = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if not np.isclose(P[i, j], 0.0):
                graphe[i].append(j)

    # Vérification de l'accessibilité pour chaque état non-absorbant
    for etat in range(n):
        if etat in absorbants:
            continue  # Pas besoin de vérifier les absorbants

        visite = set()
        file = [etat]
        trouve = False

        while file:
            courant = file.pop(0)

            if courant in absorbants:
                trouve = True
                break

            for voisin in graphe[courant]:
                if voisin not in visite:
                    visite.add(voisin)
                    file.append(voisin)

        if not trouve:
            return False

    return True
