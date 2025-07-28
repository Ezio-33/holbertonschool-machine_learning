#!/usr/bin/env python3
"""
Module pour analyser un DataFrame pandas et calculer des statistiques
descriptives.
"""


def analyze(df):
    """
    Calcule les statistiques descriptives pour toutes les colonnes sauf
    Timestamp.

    Args:
        df (pd.DataFrame): DataFrame à analyser.

    Returns:
        pd.DataFrame: DataFrame contenant les statistiques descriptives.
    """
    # Créer une copie du DataFrame
    df_copy = df.copy()

    # Supprimer la colonne Timestamp si elle existe
    if 'Timestamp' in df_copy.columns:
        df_copy = df_copy.drop('Timestamp', axis=1)

    # Calculer les statistiques descriptives
    stats = df_copy.describe()

    return stats
