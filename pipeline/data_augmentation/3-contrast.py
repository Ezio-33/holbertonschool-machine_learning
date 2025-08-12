#!/usr/bin/env python3
"""
Randomly adjust image contrast.
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjust the contrast of an image.

    Args:
        image: A 3D tf.Tensor input image to adjust the contrast.
        lower: float, lower bound for random contrast factor.
        upper: float, upper bound for random contrast factor.

    Returns:
        A 3D tf.Tensor of the contrast-adjusted image (float32 in [0, 1]).
    """
    img = tf.image.convert_image_dtype(image, tf.float32)
    factor = tf.random.uniform(shape=[], minval=lower, maxval=upper, dtype=tf.float32)
    return tf.image.adjust_contrast(img, factor)
