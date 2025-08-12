#!/usr/bin/env python3
"""
Rotate image by 90 degrees counter-clockwise.
"""
import tensorflow as tf


def rotate_image(image):
    """Rotate an image by 90 degrees counter-clockwise.

    Args:
        image: A 3D tf.Tensor H x W x C representing the image to rotate.

    Returns:
        A 3D tf.Tensor, rotated 90 degrees counter-clockwise.
    """
    return tf.image.rot90(image, k=1)
