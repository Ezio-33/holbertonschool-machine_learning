#!/usr/bin/env python3
"""
fonction qui ajoute deux tableaux par élément
"""


def add_arrays(arr1, arr2):
    """
    Ajoute deux tableaux élément par élément.

    Args:
        arr1 (list): Le premier tableau.
        arr2 (list): Le deuxième tableau.

    Return:
         Un nouveau tableau contenant la somme des éléments,
         None si les tableaux ont des tailles différentes.
    """
# Vérifier si les tableaux ont la même taille
    if len(arr1) != len(arr2):
        return None

    # Créer un nouveau tableau pour stocker les résultats
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    return result
