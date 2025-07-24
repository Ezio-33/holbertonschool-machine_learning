#!/usr/bin/env python3
"""
Module pour convertir des colonnes d'un DataFrame pandas en tableau numpy.
"""

import pandas as pd


def array(df):
    """
    Sélectionne les 10 dernières lignes des colonnes High et Close
    et les convertit en numpy.ndarray.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes High et Close.

    Returns:
        numpy.ndarray: Tableau numpy avec les 10 dernières valeurs High et
        Close.
    """
    # Sélectionner les 10 dernières lignes des colonnes High et Close
    last_10_rows = df[['High', 'Close']].tail(10)

    # Convertir en numpy array
    return last_10_rows.to_numpy()
