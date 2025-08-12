#!/usr/bin/env python3
"""
4-brightness.py

Modifie aléatoirement la luminosité d'une image.
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Change la luminosité d'une image de façon aléatoire.

    Args:
        image: tf.Tensor 3D HxWxC représentant l'image d'entrée.
        max_delta: float, variation maximale de luminosité.

    Returns:
        tf.Tensor 3D, image avec luminosité ajustée.
    """
    brightness = tf.image.random_brightness(image, max_delta)
    return brightness
