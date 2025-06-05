#!/usr/bin/env python3
"""
Module définissant l'architecture Transformer complète :
- Positional Encoding
- Attention multi-têtes
- Encodeur / Décodeur
"""

import tensorflow as tf
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calcule les encodages positionnels pour une séquence.

    Args:
        max_seq_len (int): Longueur maximale des séquences.
        dm (int): Dimension du modèle.

    Returns:
        np.ndarray: Tableau (max_seq_len, dm) avec les encodages.
    """
    def get_angles(pos, i):
        return pos / np.power(10000, (2 * (i // 2)) / np.float32(dm))

    angle_rads = get_angles(np.arange(max_seq_len)[:, np.newaxis],
                            np.arange(dm)[np.newaxis, :])
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads


def sdp_attention(Q, K, V, mask=None):
    """
    Calcule l'attention à produit scalaire réduit.

    Args:
        Q, K, V (tf.Tensor): Query, Key, Value
        mask (tf.Tensor): Masque facultatif.

    Returns:
        tuple: (attention_output, weights)
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Couche d'attention multi-têtes.
    """

    def __init__(self, dm, h):
        super().__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        batch_size = tf.shape(Q)[0]
        Q = self.split_heads(self.Wq(Q), batch_size)
        K = self.split_heads(self.Wk(K), batch_size)
        V = self.split_heads(self.Wv(V), batch_size)

        attention, weights = sdp_attention(Q, K, V, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat = tf.reshape(attention, (batch_size, -1, self.dm))
        return self.linear(concat), weights


class EncoderBlock(tf.keras.layers.Layer):
    """
    Bloc encodeur d'un Transformer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        attn_out, _ = self.mha(x, x, x, mask)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(x + attn_out)

        ffn_out = self.dense_output(self.dense_hidden(out1))
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.layernorm2(out1 + ffn_out)


class DecoderBlock(tf.keras.layers.Layer):
    """
    Bloc décodeur d'un Transformer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super().__init__()
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

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        out1 = self.layernorm1(x + self.dropout1(attn1, training=training))

        attn2, _ = self.mha2(out1, enc_output, enc_output, padding_mask)
        out2 = self.layernorm2(out1 + self.dropout2(attn2, training=training))

        ffn = self.dense_output(self.dense_hidden(out2))
        return self.layernorm3(out2 + self.dropout3(ffn, training=training))


class Encoder(tf.keras.layers.Layer):
    """
    Encodeur complet du Transformer.
    """

    def __init__(self, N, dm, h, hidden, vocab_size, max_len, drop_rate=0.1):
        super().__init__()
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(vocab_size, dm)
        self.pos_encoding = positional_encoding(max_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.pos_encoding[:x.shape[1]]
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):
    """
    Décodeur complet du Transformer.
    """

    def __init__(self, N, dm, h, hidden, vocab_size, max_len, drop_rate=0.1):
        super().__init__()
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(vocab_size, dm)
        self.pos_encoding = positional_encoding(max_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_out, training, look_ahead_mask, padding_mask):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.pos_encoding[:x.shape[1]]
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, enc_out, training, look_ahead_mask, padding_mask)
        return x


class Transformer(tf.keras.Model):
    """
    Transformer complet : encodeur + décodeur + sortie.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        super().__init__()
        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate)
        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
            self,
            inputs,
            target,
            training,
            enc_mask,
            look_ahead_mask,
            dec_mask):
        enc_output = self.encoder(inputs, training, enc_mask)
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, dec_mask)
        return self.linear(dec_output)
