#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE

df = df.drop(columns=["Weighted_Price"])
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.rename(columns={'Timestamp': 'Date'})
df = df.set_index("Date")
df["Close"] = df["Close"].ffill()
df["High"] = df["High"].fillna(df["Close"])
df["Low"] = df["Low"].fillna(df["Close"])
df["Open"] = df["Open"].fillna(df["Close"])
df[["Volume_(BTC)", "Volume_(Currency)"]] = df[[
    "Volume_(BTC)", "Volume_(Currency)"]].fillna(value=0)
df = df.loc[df.index >= pd.Timestamp("2017-01-01")]
df = df.groupby(df.index.floor('D')).agg({
    "High": "max",
    "Low": "min",
    "Open": "mean",
    "Close": "mean",
    "Volume_(BTC)": "sum",
    "Volume_(Currency)": "sum",
})
print(df)

# plotting
ax2 = df[["High",
          "Low",
          "Open", "Close", "Volume_(BTC)",
          "Volume_(Currency)"]].plot(figsize=(6, 4), grid=False,
    color=["blue", "orange", "green", "red", "purple", "brown"])
ax2.set_title("Bitcoin daily volumes")
ax2.set_xlabel("Date")
ax2.set_ylabel("Volume")
plt.tight_layout()

plt.show()
