#!/usr/bin/env python3
"""
Module Transformer pour le projet Attention
"""

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Architecture complète du Transformer : encodeur + décodeur
    """

    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Initialise le Transformer

        Args:
            N (int): nombre de blocs
            dm (int): dimension du modèle
            h (int): nombre de têtes d’attention
            hidden (int): taille du feed-forward
            input_vocab (int): taille du vocabulaire source
            target_vocab (int): taille du vocabulaire cible
            max_seq_input (int): longueur max séquence source
            max_seq_target (int): longueur max séquence cible
            drop_rate (float): taux de dropout
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)

        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)

        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training,
             encoder_mask, look_ahead_mask, decoder_mask):
        """
        Propagation avant du Transformer

        Args:
            inputs (tf.Tensor): entrée source (batch, input_seq_len)
            target (tf.Tensor): séquence cible (batch, target_seq_len)
            training (bool): mode entraînement
            encoder_mask (tf.Tensor): masque sur l’entrée
            look_ahead_mask (tf.Tensor): masque anti-futur pour cible
            decoder_mask (tf.Tensor): masque de padding pour cible

        Returns:
            tf.Tensor: logits (batch, target_seq_len, target_vocab)
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)

        decoder_output = self.decoder(target, encoder_output,
                                      training, look_ahead_mask,
                                      decoder_mask)

        final_output = self.linear(decoder_output)

        return final_output
