#!/usr/bin/env python3
"""
Random crop of an image using TensorFlow ops.
"""
import tensorflow as tf


def crop_image(image, size):
    """Perform a random crop of an image.

    Args:
        image: A 3D tf.Tensor H x W x C representing the image to crop.
        size: Tuple[int, int, int], target crop size
            as (height, width, channels).

    Returns:
        A 3D tf.Tensor of the cropped image with shape given by size.
    """
    return tf.image.random_crop(image, size)
