#!/usr/bin/env python3
"""
Convertit un modèle Word2Vec Gensim en couche Keras Embedding.
"""

import tensorflow as tf


def gensim_to_keras(model):
    """
    Transforme un modèle Gensim Word2Vec en couche Embedding Keras.

    Args:
        model (gensim.models.Word2Vec): modèle Gensim entraîné

    Returns:
        tf.keras.layers.Embedding: couche embedding avec les poids du modèle
    """
    keyed_vectors = model.wv
    weights = keyed_vectors.vectors

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return embedding_layer
