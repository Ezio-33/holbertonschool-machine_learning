#!/usr/bin/env python3
"""
Module pour inverser et transposer un DataFrame pandas.
"""

import pandas as pd


def flip_switch(df):
    """
    Trie les données en ordre chronologique inverse et transpose le DataFrame.

    Args:
        df (pd.DataFrame): DataFrame à transformer.

    Returns:
        pd.DataFrame: DataFrame trié en ordre inverse et transposé.
    """
    # Trier les données en ordre chronologique inverse (du plus récent au plus
    # ancien) On utilise sort_index avec ascending=False pour inverser l'ordre
    # des lignes
    df_sorted = df.sort_index(ascending=False)

    # Transposer le DataFrame (lignes deviennent colonnes et vice versa)
    df_transposed = df_sorted.T

    return df_transposed
