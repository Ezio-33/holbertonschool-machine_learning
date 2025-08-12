#!/usr/bin/env python3
"""
Randomly change image brightness.
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly change the brightness of an image.

    Args:
        image: A 3D tf.Tensor input image to change.
        max_delta: float, maximum delta to add/subtract to brightness.

    Returns:
        A 3D tf.Tensor with altered brightness (float32 in [0, 1]).
    """
    img = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.random_brightness(img, max_delta=max_delta)
