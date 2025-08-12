#!/usr/bin/env python3
"""
5-hue.py

Modifie la teinte (hue) d'une image.
"""
import tensorflow as tf


def change_hue(image, delta):
    """Change la teinte (hue) d'une image d'un delta donné.

    Args:
        image: tf.Tensor 3D HxWxC représentant l'image d'entrée.
        delta: float, quantité de décalage de la teinte.

    Returns:
        tf.Tensor 3D, image avec teinte ajustée.
    """
    hue = tf.image.adjust_hue(image, delta)
    return hue
