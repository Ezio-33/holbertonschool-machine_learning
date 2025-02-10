#!/usr/bin/env python3
"""Module implémentant LeNet-5 avec TensorFlow 1.x"""

import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Construit LeNet-5 modifié pour MNIST

    Args:
        x: Images d'entrée (m, 28, 28, 1)
        y: Labels one-hot (m, 10)

    Returns:
        y_pred: Prédictions (softmax)
        train_op: Opération d'entraînement Adam
        loss: Fonction de perte
        acc: Précision
    """
    # Initialisation He pour les poids
    init = tf.contrib.layers.variance_scaling_initializer()

    # Première couche convolutive (6 filtres 5x5)
    conv1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=init)(x)

    # Premier pooling (2x2)
    pool1 = tf.layers.MaxPooling2D(
        pool_size=[2, 2],
        strides=2)(conv1)

    # Deuxième couche convolutive (16 filtres 5x5)
    conv2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=init)(pool1)

    # Deuxième pooling (2x2)
    pool2 = tf.layers.MaxPooling2D(
        pool_size=[2, 2],
        strides=2)(conv2)

    # Aplatissement pour les couches denses
    flat = tf.layers.Flatten()(pool2)

    # Première couche dense (120 neurones)
    dense1 = tf.layers.Dense(
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=init)(flat)

    # Deuxième couche dense (84 neurones)
    dense2 = tf.layers.Dense(
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=init)(dense1)

    # Couche de sortie (10 classes)
    logits = tf.layers.Dense(
        units=10,
        kernel_initializer=init)(dense2)

    # Activation softmax séparée
    y_pred = tf.nn.softmax(logits)

    # Calcul de la perte (cross-entropy)
    loss = tf.losses.softmax_cross_entropy(y, logits)

    # Optimisation avec Adam
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Calcul de la précision
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    return y_pred, train_op, loss, acc
