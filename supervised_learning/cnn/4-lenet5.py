#!/usr/bin/env python3
"""Architecture LeNet-5 avec TensorFlow 1.x"""

import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Construit l'architecture LeNet-5 modifiée
    Args:
        x: Placeholder (m, 28, 28, 1) - Images d'entrée
        y: Placeholder (m, 10) - Labels one-hot

    Returns:
        y_pred: Tenseur de prédiction
        train_op: Opération d'entraînement Adam
        loss: Tenseur de perte
        acc: Tenseur de précision
    """
    init = tf.keras.initializers.VarianceScaling(
        scale=2.0)  # Initialisation He

    # Couche 1 : Conv2D -> ReLU -> MaxPool
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=init
    )
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=(2, 2),
        strides=(2, 2)
    )

    # Couche 2 : Conv2D -> ReLU -> MaxPool
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=init
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=(2, 2),
        strides=(2, 2)
    )

    # Aplatissement
    flat = tf.layers.flatten(pool2)

    # Couche 3 : Dense -> ReLU
    dense1 = tf.layers.dense(
        inputs=flat,
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=init
    )

    # Couche 4 : Dense -> ReLU
    dense2 = tf.layers.dense(
        inputs=dense1,
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=init
    )

    # Couche de sortie : Dense + Softmax
    y_pred = tf.layers.dense(
        inputs=dense2,
        units=10,
        activation=tf.nn.softmax,
        kernel_initializer=init
    )

    # Calcul de la perte
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y, y_pred))

    # Optimiseur Adam
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Calcul de la précision
    acc = tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(y, 1),
        tf.argmax(y_pred, 1)
    ), tf.float32))

    return y_pred, train_op, loss, acc
