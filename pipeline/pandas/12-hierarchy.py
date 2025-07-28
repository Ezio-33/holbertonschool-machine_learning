#!/usr/bin/env python3
"""
Module pour créer une hiérarchie avec Timestamp comme premier niveau.
"""

import pandas as pd


def hierarchy(df1, df2):
    """
    Réorganise le MultiIndex pour que Timestamp soit le premier niveau
    et filtre une plage de timestamps spécifique.

    Args:
        df1 (pd.DataFrame): Premier DataFrame (coinbase).
        df2 (pd.DataFrame): Deuxième DataFrame (bitstamp).

    Returns:
        pd.DataFrame: DataFrame avec MultiIndex réorganisé et données filtrées.
    """
    # Importer la fonction index de la tâche 10
    index = __import__('10-index').index

    # 1. Indexer les deux DataFrames sur leurs colonnes Timestamp
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # 2. Filtrer les deux DataFrames pour la plage de timestamps spécifiée
    # Timestamps de 1417411980 à 1417417980, inclusive
    start_timestamp = 1417411980
    end_timestamp = 1417417980

    df1_filtered = df1_indexed[(df1_indexed.index >= start_timestamp) &
                               (df1_indexed.index <= end_timestamp)]
    df2_filtered = df2_indexed[(df2_indexed.index >= start_timestamp) &
                               (df2_indexed.index <= end_timestamp)]

    # 3. Concaténer avec des labels
    concatenated_df = pd.concat([df2_filtered, df1_filtered],
                                keys=['bitstamp', 'coinbase'])

    # 4. Réorganiser le MultiIndex pour que Timestamp soit le premier niveau
    # Échanger les niveaux de l'index (niveau 0: exchange, niveau 1: timestamp)
    # pour obtenir (niveau 0: timestamp, niveau 1: exchange)
    reordered_df = concatenated_df.swaplevel(0, 1).sort_index()

    return reordered_df
