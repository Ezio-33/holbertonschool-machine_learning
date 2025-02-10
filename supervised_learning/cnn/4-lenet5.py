#!/usr/bin/env python3
"""Module implémentant l'architecture LeNet-5 avec TensorFlow 1"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def lenet5(x, y):
    """
    Construit l'architecture LeNet-5 modifiée

    Args:
        x: Placeholder pour les images d'entrée (m, 28, 28, 1)
        y: Placeholder pour les labels one-hot (m, 10)

    Returns:
        y_pred: Tenseur de sortie avec softmax
        train_op: Opération d'entraînement Adam
        loss: Tenseur de perte
        accuracy: Tenseur de précision
    """
    # Initialisation He pour tous les poids
    init = tf.contrib.layers.variance_scaling_initializer()

    # Première couche convolutive
    conv1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=init
    )(x)

    # Premier max pooling
    pool1 = tf.layers.MaxPooling2D(
        pool_size=[2, 2],
        strides=2
    )(conv1)

    # Deuxième couche convolutive
    conv2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=init
    )(pool1)

    # Deuxième max pooling
    pool2 = tf.layers.MaxPooling2D(
        pool_size=[2, 2],
        strides=2
    )(conv2)

    # Aplatissement
    flatten = tf.layers.Flatten()(pool2)

    # Première couche dense
    dense1 = tf.layers.Dense(
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=init
    )(flatten)

    # Deuxième couche dense
    dense2 = tf.layers.Dense(
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=init
    )(dense1)

    # Couche de sortie
    logits = tf.layers.Dense(
        units=10,
        kernel_initializer=init
    )(dense2)

    # Prédictions avec softmax
    y_pred = tf.nn.softmax(logits)

    # Calcul de la perte
    loss = tf.losses.softmax_cross_entropy(y, logits)

    # Optimisation
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Calcul de la précision
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return y_pred, train_op, loss, accuracy
