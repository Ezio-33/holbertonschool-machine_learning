#!/usr/bin/env python3
"""
Module Decoder pour le projet Attention
"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Décodeur complet du modèle Transformer
    """

    def __init__(
            self,
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_len,
            drop_rate=0.1):
        """
        Initialise les composants du décodeur Transformer

        Args:
            N (int): nombre de blocs de décodeur
            dm (int): dimension totale du modèle
            h (int): nombre de têtes
            hidden (int): taille du réseau dense
            target_vocab (int): taille du vocabulaire cible
            max_seq_len (int): taille max d’une séquence cible
            drop_rate (float): taux de dropout
        """
        super(Decoder, self).__init__()

        self.dm = dm
        self.N = N

        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Propagation avant du décodeur

        Args:
            x (tf.Tensor): séquence cible (batch, target_seq_len)
            encoder_output (tf.Tensor): sortie de l’encodeur
            (batch, input_seq_len, dm)
            training (bool): booléen pour mode entraînement
            look_ahead_mask (tf.Tensor): masque anti-futur
            padding_mask (tf.Tensor): masque de padding

        Returns:
            tf.Tensor: sortie du décodeur (batch, target_seq_len, dm)
        """
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        x += self.positional_encoding[:seq_len, :]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                training,
                look_ahead_mask,
                padding_mask)

        return x
