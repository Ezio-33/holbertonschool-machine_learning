#!/usr/bin/env python3
"""
Module pour nettoyer un DataFrame pandas en supprimant les valeurs NaN.
"""

import pandas as pd


def prune(df):
    """
    Supprime toutes les lignes où la colonne Close contient des valeurs NaN.

    Args:
        df (pd.DataFrame): DataFrame à nettoyer.

    Returns:
        pd.DataFrame: DataFrame modifié sans les lignes où Close est NaN.
    """
    # Supprimer les lignes où la colonne Close a des valeurs NaN
    cleaned_df = df.dropna(subset=['Close'])

    return cleaned_df
