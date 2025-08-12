#!/usr/bin/env python3
"""
3-contrast.py

Ajuste aléatoirement le contraste d'une image.
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Change le contraste d'une image de façon aléatoire.

    Args:
        image: tf.Tensor 3D HxWxC représentant l'image d'entrée.
        lower: float, borne basse du facteur de contraste.
        upper: float, borne haute du facteur de contraste.

    Returns:
        tf.Tensor 3D, image avec contraste ajusté.
    """
    contrast = tf.image.random_contrast(image, lower, upper)
    return contrast
