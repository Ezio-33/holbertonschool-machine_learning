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
    # Étape 1 : Produit scalaire entre Q et K transposé
    # (..., seq_len_q, seq_len_v)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Étape 2 : Mise à l’échelle
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_scores = matmul_qk / tf.math.sqrt(dk)

    # Étape 3 : Application du masque si fourni
    if mask is not None:
        scaled_scores += (mask * -1e9)

    # Étape 4 : Calcul des poids d’attention
    weights = tf.nn.softmax(scaled_scores, axis=-1)

    # Étape 5 : Calcul de la sortie par moyenne pondérée
    output = tf.matmul(weights, V)

    return output, weights
