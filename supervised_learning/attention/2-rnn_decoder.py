#!/usr/bin/env python3
"""
Module RNNDecoder pour le projet Attention
"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Classe RNNDecoder pour décodeur à base de GRU avec attention
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialise les couches du décodeur

        Args:
            vocab (int): taille du vocabulaire de sortie
            embedding (int): dimension des vecteurs d'embedding
            units (int): nombre d’unités cachées de la GRU
            batch (int): taille du batch
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Propagation avant du décodeur avec attention

        Args:
            x (tf.Tensor): mot précédent (batch, 1)
            s_prev (tf.Tensor): état caché précédent (batch, units)
            hidden_states (tf.Tensor): sorties de l’encodeur
            (batch, input_seq_len, units)

        Returns:
            y (tf.Tensor): sortie vocabulaire (batch, vocab)
            s (tf.Tensor): nouvel état caché (batch, units)
        """
        x = self.embedding(x)

        context, _ = self.attention(s_prev, hidden_states)

        context = tf.expand_dims(context, 1)

        x = tf.concat([context, x], axis=-1)

        x, s = self.gru(x)

        x = tf.reshape(x, (-1, x.shape[2]))

        y = self.F(x)

        return y, s
