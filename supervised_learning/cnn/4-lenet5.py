#!/usr/bin/env python3
"""Module implémentant l'architecture LeNet-5 modifiée"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def lenet5(x, y):
    """
    Construit le réseau LeNet-5 pour reconnaître les chiffres

    Args:
        x: Images d'entrée (m, 28, 28, 1)
        y: Labels attendus (m, 10)
    """
    # Initialisation spéciale pour les poids
    init = tf.contrib.layers.variance_scaling_initializer()

    # Première couche: 6 filtres de 5x5
    conv1 = tf.layers.Conv2D(
        filters=6,          # Nombre de loupes différentes
        kernel_size=5,      # Taille de chaque loupe
        padding='same',     # On garde les bords
        activation=tf.nn.relu,  # On garde que les valeurs positives
        kernel_initializer=init
    )(x)

    # Réduction de taille par 2
    pool1 = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    )(conv1)

    # Deuxième couche: 16 filtres de 5x5
    conv2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding='valid',    # On ne garde pas les bords
        activation=tf.nn.relu,
        kernel_initializer=init
    )(pool1)

    # Nouvelle réduction de taille
    pool2 = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    )(conv2)

    # On met tout à plat
    flatten = tf.layers.Flatten()(pool2)

    # Première couche dense: 120 neurones
    dense1 = tf.layers.Dense(
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=init
    )(flatten)

    # Deuxième couche dense: 84 neurones
    dense2 = tf.layers.Dense(
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=init
    )(dense1)

    # Couche de sortie: 10 neurones (un par chiffre)
    logits = tf.layers.Dense(
        units=10,
        kernel_initializer=init
    )(dense2)

    # Probabilités pour chaque chiffre
    y_pred = tf.nn.softmax(logits)

    # Calcul de l'erreur
    loss = tf.losses.softmax_cross_entropy(y, logits)

    # Optimisation pour réduire l'erreur
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Calcul de la précision
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return y_pred, train_op, loss, accuracy
