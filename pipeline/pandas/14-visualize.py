#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE

# 1. Supprimer la colonne Weighted_Price
df = df.drop('Weighted_Price', axis=1)

# 2. Renommer la colonne Timestamp en Date
df = df.rename(columns={'Timestamp': 'Date'})

# 3. Convertir les valeurs timestamp en valeurs de date
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# 4. Indexer le DataFrame sur Date
df = df.set_index('Date')

# 5. Remplir les valeurs manquantes selon les règles spécifiées
# Close: valeur de la ligne précédente
df['Close'] = df['Close'].ffill()

# High, Low, Open: valeur Close de la même ligne
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

# Volume_(BTC) et Volume_(Currency): 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# 6. Filtrer les données de 2017 et au-delà
df = df[df.index >= '2017-01-01']

# 7. Regrouper par jour avec les agrégations spécifiées
df_daily = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

print(df_daily)

# Créer la visualisation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Graphique 1: Prix (OHLC)
ax1.plot(df_daily.index, df_daily['Open'], label='Open', alpha=0.7)
ax1.plot(df_daily.index, df_daily['High'], label='High', alpha=0.7)
ax1.plot(df_daily.index, df_daily['Low'], label='Low', alpha=0.7)
ax1.plot(df_daily.index, df_daily['Close'], label='Close', alpha=0.7)

ax1.set_title('Bitcoin Price Data (2017-2019)')
ax1.set_ylabel('Price (USD)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Graphique 2: Volume
ax2.plot(
    df_daily.index,
    df_daily['Volume_(BTC)'],
    label='Volume (BTC)',
    alpha=0.7)
ax2_twin = ax2.twinx()
ax2_twin.plot(
    df_daily.index,
    df_daily['Volume_(Currency)'],
    label='Volume (Currency)',
    color='orange',
    alpha=0.7)

ax2.set_title('Trading Volume')
ax2.set_ylabel('Volume (BTC)')
ax2_twin.set_ylabel('Volume (Currency)')
ax2.set_xlabel('Date')
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Retourner le DataFrame transformé
df = df_daily
