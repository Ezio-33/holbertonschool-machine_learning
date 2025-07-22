#!/usr/bin/env python3
"""
Module pour convertir un tableau NumPy en DataFrame pandas.
"""

import pandas as pd


def from_numpy(array):
    """
    Crée un pd.DataFrame à partir d'un np.ndarray.

    Args:
        array (np.ndarray): Tableau NumPy à convertir.

    Returns:
        pd.DataFrame: DataFrame avec des colonnes nommées de A à Z.
    """
    # Déterminer le nombre de colonnes
    num_cols = array.shape[1] if len(array.shape) > 1 else 1

    # Générer les noms des colonnes (A, B, C, ...)
    column_names = [chr(65 + i) for i in range(num_cols)]

    # Créer le DataFrame avec les colonnes nommées
    df = pd.DataFrame(array, columns=column_names)

    return df
