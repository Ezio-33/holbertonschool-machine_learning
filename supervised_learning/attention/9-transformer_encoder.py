#!/usr/bin/env python3
"""
Module Encoder pour le projet Attention
"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Encodeur complet d’un modèle Transformer
    """

    def __init__(
            self,
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_len,
            drop_rate=0.1):
        """
        Initialise les composants de l’encodeur Transformer

        Args:
            N (int): nombre de blocs d’encodeur
            dm (int): dimension totale du modèle
            h (int): nombre de têtes d’attention
            hidden (int): taille de la couche feed-forward
            input_vocab (int): taille du vocabulaire d’entrée
            max_seq_len (int): taille maximale d’une séquence
            drop_rate (float): taux de dropout
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Propagation avant de l’encodeur

        Args:
            x (tf.Tensor): séquence d’entrée (batch, input_seq_len)
            training (bool): si True, mode entraînement
            mask (tf.Tensor): masque d’attention

        Returns:
            tf.Tensor: sortie de l’encodeur (batch, input_seq_len, dm)
        """
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)

        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        x += self.positional_encoding[:seq_len, :]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        return x
