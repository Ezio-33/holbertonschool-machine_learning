#!/usr/bin/env python3
"""
Module pour concaténer deux DataFrames pandas avec des conditions spécifiques.
"""
import pandas as pd


def concat(df1, df2):
    """
    Concatène deux DataFrames en appliquant des conditions et des labels
    spécifiques.

    Args:
        df1 (pd.DataFrame): Premier DataFrame (coinbase).
        df2 (pd.DataFrame): Deuxième DataFrame (bitstamp).

    Returns:
        pd.DataFrame: DataFrame concaténé avec labels et conditions appliquées.
    """
    # Importer la fonction index de la tâche 10
    index = __import__('10-index').index

    # 1. Indexer les deux DataFrames sur leurs colonnes Timestamp
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # 2. Filtrer df2 pour inclure seulement les timestamps <= 1417411920
    df2_filtered = df2_indexed[df2_indexed.index <= 1417411920]

    # 3. Concaténer avec des labels (df2 en haut, df1 en bas)
    concatenated_df = pd.concat([df2_filtered, df1_indexed],
                                keys=['bitstamp', 'coinbase'])

    return concatenated_df
