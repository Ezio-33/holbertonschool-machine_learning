#!/usr/bin/env python3
"""
Module d'encodage RNN pour le projet Attention
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    Classe RNNEncoder qui encode une séquence d’entrée à l’aide d’un GRU.
    Hérite de tf.keras.layers.Layer.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialise les attributs de l'encodeur RNN.

        Args:
            vocab (int): taille du vocabulaire.
            embedding (int): dimension des vecteurs d'embedding.
            units (int): nombre d'unités cachées dans la GRU.
            batch (int): taille du batch.
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """
        Initialise l’état caché à des zéros.

        Returns:
            tf.Tensor: tenseur de zéros de forme (batch, units)
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Exécute la propagation avant de l’encodeur.

        Args:
            x (tf.Tensor): séquence d’indices de mots (batch, input_seq_len)
            initial (tf.Tensor): état caché initial (batch, units)

        Returns:
            outputs (tf.Tensor): sorties GRU (batch, input_seq_len, units)
            hidden (tf.Tensor): dernier état caché (batch, units)
        """
        x_embed = self.embedding(x)
        outputs, hidden = self.gru(x_embed, initial_state=initial)
        return outputs, hidden
