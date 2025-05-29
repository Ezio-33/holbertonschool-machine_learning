#!/usr/bin/env python3
"""
Module EncoderBlock pour le projet Attention
"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Bloc d’encodeur pour un Transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialise les couches du bloc encodeur

        Args:
            dm (int): dimension du modèle
            h (int): nombre de têtes
            hidden (int): taille de la couche feed-forward cachée
            drop_rate (float): taux de dropout
        """
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Propagation avant du bloc encodeur

        Args:
            x (tf.Tensor): entrée (batch, input_seq_len, dm)
            training (bool): mode entraînement ou non
            mask (tf.Tensor): masque optionnel

        Returns:
            tf.Tensor: sortie (batch, input_seq_len, dm)
        """
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
