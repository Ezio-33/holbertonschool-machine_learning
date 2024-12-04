#!/usr/bin/env python3

def add_arrays(arr1, arr2):
    # Vérifier si les tableaux ont la même taille
    if len(arr1) != len(arr2):
        return None

    # Créer un nouveau tableau pour stocker les résultats
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    return result
