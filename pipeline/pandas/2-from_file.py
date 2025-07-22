#!/usr/bin/env python3
"""
Module pour charger des données depuis un fichier en DataFrame pandas.
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Charge des données depuis un fichier en pd.DataFrame.

    Args:
        filename (str): Le nom du fichier à charger.
        delimiter (str): Le séparateur de colonnes.

    Returns:
        pd.DataFrame: DataFrame contenant les données du fichier.
    """
    return pd.read_csv(filename, delimiter=delimiter)
