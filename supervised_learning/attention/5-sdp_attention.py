#!/usr/bin/env python3
"""
Module pour calculer l'attention Scaled Dot Product
dans les architectures Transformer
"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calcule l’attention par produit scalaire mis à l’échelle.

    Args:
        Q (tf.Tensor): requêtes (..., seq_len_q, dk)
        K (tf.Tensor): clés (..., seq_len_v, dk)
        V (tf.Tensor): valeurs (..., seq_len_v, dv)
        mask (tf.Tensor | None): masque optionnel

    Returns:
        output (tf.Tensor): résultat (..., seq_len_q, dv)
        weights (tf.Tensor): poids (..., seq_len_q, seq_len_v)
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_scores = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_scores += (mask * -1e9)

    weights = tf.nn.softmax(scaled_scores, axis=-1)

    output = tf.matmul(weights, V)

    return output, weights
