#!/usr/bin/env python3
"""
Module de prétraitement optimisé des données BTC pour modèle RNN
"""

import pandas as pd
import numpy as np


def preprocess():
    """
    Prétraitement optimisé des données pour économiser la mémoire
    """
    # Charger les fichiers CSV
    coinbase = pd.read_csv("coinbase.csv")
    bitstamp = pd.read_csv("bitstamp.csv")

    # Fusion et moyenne des prix
    data = pd.merge(coinbase, bitstamp, on='Timestamp',
                    suffixes=('_coinbase', '_bitstamp'))
    data['Close'] = data[['Close_coinbase', 'Close_bitstamp']].mean(axis=1)

    # Garder seulement le Timestamp et la fermeture
    data = data[['Timestamp', 'Close']].dropna()

    # Échantillonnage horaire pour réduire les données
    data = data.iloc[::60].reset_index(drop=True)

    # Normalisation entre 0 et 1
    min_val = data['Close'].min()
    max_val = data['Close'].max()
    data['Close'] = (data['Close'] - min_val) / (max_val - min_val)

    # Sauvegarde intermédiaire propre
    data.to_csv("btc_hourly.csv", index=False)

    # Création des séquences avec fenêtrage glissant via générateur
    def generate_sequences(values, window_size=24):
        for i in range(len(values) - window_size):
            yield values[i:i + window_size], values[i + window_size]

    sequences_gen = generate_sequences(data['Close'].values, window_size=24)

    # Sauvegarde des séquences dans un fichier numpy optimisé
    sequences, targets = [], []
    for seq, target in sequences_gen:
        sequences.append(seq)
        targets.append(target)

    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    np.savez_compressed("btc_preprocessed.npz",
                        sequences=sequences, targets=targets)


if __name__ == "__main__":
    preprocess()
