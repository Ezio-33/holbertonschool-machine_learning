#!/usr/bin/env python3
"""
Module pour renommer les colonnes d'un DataFrame pandas.
"""

import pandas as pd


def rename(df):
    """
    Renomme la colonne Timestamp en Datetime, convertit les valeurs en
    datetime, et retourne seulement les colonnes Datetime et Close.

    Args:
        df (pd.DataFrame): DataFrame contenant une colonne Timestamp.

    Returns:
        pd.DataFrame: DataFrame modifié avec colonnes Datetime et Close
        seulement.
    """
    # Créer une copie du DataFrame
    df_copy = df.copy()

    # Renommer la colonne Timestamp en Datetime
    df_copy = df_copy.rename(columns={'Timestamp': 'Datetime'})

    # Convertir les valeurs timestamp en datetime
    df_copy['Datetime'] = pd.to_datetime(df_copy['Datetime'], unit='s')

    # Retourner seulement les colonnes Datetime et Close
    return df_copy[['Datetime', 'Close']]
