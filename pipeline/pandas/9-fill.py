#!/usr/bin/env python3
"""
Module pour remplir les valeurs manquantes dans un DataFrame pandas.
"""


def fill(df):
    """
    Supprime la colonne Weighted_Price et remplit les valeurs manquantes
    selon des règles spécifiques.

    Args:
        df (pd.DataFrame): DataFrame à modifier.

    Returns:
        pd.DataFrame: DataFrame modifié avec valeurs manquantes remplies.
    """
    # Créer une copie du DataFrame
    df_copy = df.copy()

    # 1. Supprimer la colonne Weighted_Price
    df_copy = df_copy.drop('Weighted_Price', axis=1)

    # 2. Remplir les valeurs manquantes dans Close avec la valeur de la ligne
    # précédente
    df_copy['Close'] = df_copy['Close'].ffill()

    # 3. Remplir les valeurs manquantes dans High, Low, et Open avec la valeur
    # Close correspondante
    df_copy['High'] = df_copy['High'].fillna(df_copy['Close'])
    df_copy['Low'] = df_copy['Low'].fillna(df_copy['Close'])
    df_copy['Open'] = df_copy['Open'].fillna(df_copy['Close'])

    # 4. Remplir les valeurs manquantes dans Volume_(BTC) et Volume_(Currency)
    # avec 0
    df_copy['Volume_(BTC)'] = df_copy['Volume_(BTC)'].fillna(0)
    df_copy['Volume_(Currency)'] = df_copy['Volume_(Currency)'].fillna(0)

    return df_copy
