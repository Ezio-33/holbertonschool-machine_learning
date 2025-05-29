#!/usr/bin/env python3
"""
Module DecoderBlock pour le projet Attention
"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Bloc de décodeur Transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialise les couches du bloc décodeur

        Args:
            dm (int): dimension du modèle
            h (int): nombre de têtes
            hidden (int): taille de la couche feed-forward cachée
            drop_rate (float): taux de dropout
        """
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self,
             x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Propagation avant du bloc décodeur

        Args:
            x (tf.Tensor): entrée cible (batch, target_seq_len, dm)
            encoder_output (tf.Tensor):
            sortie encodeur (batch, input_seq_len,dm)
            training (bool): mode entraînement ou inférence
            look_ahead_mask (tf.Tensor): masque anti-futur
            padding_mask (tf.Tensor): masque des tokens vides

        Returns:
            tf.Tensor: sortie (batch, target_seq_len, dm)
        """
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn = self.dense_hidden(out2)
        ffn = self.dense_output(ffn)
        ffn = self.dropout3(ffn, training=training)
        out3 = self.layernorm3(out2 + ffn)

        return out3
