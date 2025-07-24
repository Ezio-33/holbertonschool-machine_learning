#!/usr/bin/env python3
"""
Module pour découper un DataFrame pandas selon des critères spécifiques.
"""


def slice(df):
    """
    Extrait les colonnes High, Low, Close, et Volume_(BTC)
    et sélectionne chaque 60ème ligne.

    Args:
        df (pd.DataFrame): DataFrame à découper.

    Returns:
        pd.DataFrame: DataFrame découpé avec les colonnes et lignes
        sélectionnées.
    """
    # Extraire les colonnes spécifiées
    columns_to_extract = ['High', 'Low', 'Close', 'Volume_(BTC)']

    # Sélectionner chaque 60ème ligne (index 0, 60, 120, 180, ...)
    sliced_df = df[columns_to_extract][::60]

    return sliced_df
