#!/usr/bin/env python3
"""
Module pour trier un DataFrame pandas par la colonne High.
"""

import pandas as pd


def high(df):
    """
    Trie le DataFrame par la colonne High en ordre décroissant.

    Args:
        df (pd.DataFrame): DataFrame à trier.

    Returns:
        pd.DataFrame: DataFrame trié par High en ordre décroissant.
    """
    # Trier par la colonne High en ordre décroissant
    sorted_df = df.sort_values(by='High', ascending=False)

    return sorted_df
