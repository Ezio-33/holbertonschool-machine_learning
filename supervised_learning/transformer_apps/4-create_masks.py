#!/usr/bin/env python3
"""Transformer Application - Création des masques"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    Crée les masques pour l'encodeur et le décodeur d'un Transformer.

    Args:
        inputs (tf.Tensor): séquences d'entrée (batch_size, seq_len_in)
        target (tf.Tensor): séquences cibles (batch_size, seq_len_out)

    Returns:
        encoder_mask: tf.Tensor (batch_size, 1, 1, seq_len_in)
        combined_mask: tf.Tensor (batch_size, 1, seq_len_out, seq_len_out)
        decoder_mask: tf.Tensor (batch_size, 1, 1, seq_len_in)
    """
    enc_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    enc_mask = enc_mask[:, tf.newaxis, tf.newaxis, :]

    dec_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    dec_mask = dec_mask[:, tf.newaxis, tf.newaxis, :]

    size = target.shape[1]

    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    look_ahead_mask = tf.cast(look_ahead_mask, tf.float32)

    dec_target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_target_mask = dec_target_mask[:, tf.newaxis, tf.newaxis, :]

    combined_mask = tf.maximum(dec_target_mask, look_ahead_mask)

    return enc_mask, combined_mask, dec_mask
