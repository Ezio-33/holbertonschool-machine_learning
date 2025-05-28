#!/usr/bin/env python3
"""
Module MultiHeadAttention pour le projet Attention
"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Classe MultiHeadAttention : calcule l’attention multi-tête
    """

    def __init__(self, dm, h):
        """
        Initialise les couches de la multi-head attention.

        Args:
            dm (int): dimension du modèle.
            h (int): nombre de têtes.
        """
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def reshape_tensor(self, x, h, forward=True):
        """
        Réorganise le tenseur pour appliquer/recomposer les têtes.

        Args:
            x (tf.Tensor): le tenseur à transformer
            h (int): nombre de têtes
            forward (bool): True pour préparer, False pour reconstruire

        Returns:
            tf.Tensor: le tenseur réorganisé
        """
        if forward:
            x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], h, -1))
            return tf.transpose(x, perm=(0, 2, 1, 3))
        else:
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            return tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], self.dm))

    def call(self, Q, K, V, mask=None):
        """
        Applique la multi-head attention sur Q, K, V.

        Args:
            Q, K, V: Tenseurs d'entrée (batch, seq_len, dm)
            mask: masque d’attention optionnel

        Returns:
            output (tf.Tensor): résultat de l’attention (batch, seq_len, dm)
            weights (tf.Tensor): poids d’attention
            (batch, h, seq_len_q, seq_len_v)
        """
        # Projection dans l’espace commun
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # Préparer pour multi-head : (batch, h, seq_len, depth)
        Q = self.reshape_tensor(Q, self.h, True)
        K = self.reshape_tensor(K, self.h, True)
        V = self.reshape_tensor(V, self.h, True)

        # Attention multi-tête
        # out: (batch, h, seq_len_q, depth)
        out, weights = sdp_attention(Q, K, V, mask)

        # Fusion des têtes : (batch, seq_len_q, dm)
        out = self.reshape_tensor(out, self.h, False)

        # Projection finale
        return self.linear(out), weights
