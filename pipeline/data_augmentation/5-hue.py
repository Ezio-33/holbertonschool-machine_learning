#!/usr/bin/env python3
"""
Change image hue.
"""
import tensorflow as tf


def change_hue(image, delta):
    """Change the hue of an image by a fixed delta.

    Args:
        image: A 3D tf.Tensor input image to change.
        delta: float, amount to shift hue.

    Returns:
        A 3D tf.Tensor with hue adjusted (float32 in [0, 1]).
    """
    img = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.adjust_hue(img, delta)
