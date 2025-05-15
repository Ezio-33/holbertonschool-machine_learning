#!/usr/bin/env python3
"""
Module optimisé pour entraîner et évaluer un modèle RNN
"""

import numpy as np
import tensorflow as tf


def main():
    """
    Charge données prétraitées, entraîne et évalue le modèle
    """
    data = np.load("btc_preprocessed.npz")
    X, y = data['sequences'], data['targets']

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Ajustement pour ajouter la dimension requise par TensorFlow
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(24, 1)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(train_ds, epochs=5, validation_data=test_ds)

    loss = model.evaluate(test_ds)
    print("Erreur finale (MSE):", loss)


if __name__ == "__main__":
    main()
