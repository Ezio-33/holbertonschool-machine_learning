#!/usr/bin/env python3
"""
Flip image horizontally using TensorFlow ops.
"""
import tensorflow as tf


def flip_image(image):
    """Flip an image horizontally (left-right).

    Args:
        image: A 3D tf.Tensor H x W x C representing the image to flip.

    Returns:
        A 3D tf.Tensor, horizontally flipped copy of the input image.
    """
    return tf.image.flip_left_right(image)
