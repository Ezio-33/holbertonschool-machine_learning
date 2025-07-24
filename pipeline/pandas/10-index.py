#!/usr/bin/env python3
"""
Module pour définir l'index d'un DataFrame pandas.
"""


def index(df):
    """
    Définit la colonne Timestamp comme index du DataFrame.

    Args:
        df (pd.DataFrame): DataFrame à modifier.

    Returns:
        pd.DataFrame: DataFrame modifié avec Timestamp comme index.
    """
    # Définir la colonne Timestamp comme index
    df_indexed = df.set_index('Timestamp')

    return df_indexed
