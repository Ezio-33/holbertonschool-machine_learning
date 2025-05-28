#!/usr/bin/env python3
"""
Module SelfAttention pour le projet Attention
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Classe SelfAttention qui calcule
        l’attention pour la traduction automatique.
    """

    def __init__(self, units):
        """
        Initialise la couche SelfAttention.

        Args:
            units (int): nombre d’unités dans le modèle d’alignement.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Calcule le vecteur de contexte et les poids d’attention.

        Args:
            s_prev (tf.Tensor): état caché précédent du décodeur (batch, units)
            hidden_states (tf.Tensor): sorties de
                        l’encodeur (batch, input_seq_len, units)

        Returns:
            context (tf.Tensor): vecteur de contexte (batch, units)
            weights (tf.Tensor): poids d’attention (batch, input_seq_len, 1)
        """
        # On rajoute un axe pour pouvoir le combiner avec hidden_states
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # On applique W et U
        score = self.V(
            tf.nn.tanh(
                self.W(s_prev_expanded) +
                self.U(hidden_states)))

        # Poids d’attention via softmax sur chaque séquence
        weights = tf.nn.softmax(score, axis=1)

        # Vecteur de contexte : somme pondérée des hidden_states
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
