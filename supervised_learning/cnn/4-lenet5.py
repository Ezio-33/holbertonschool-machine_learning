#!/usr/bin/env python3
"""LeNet-5 avec initialisation He et gestion correcte des logits"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def lenet5(x, y):
    """
    Construit un modèle LeNet-5 modifié en utilisant TensorFlow.
    Args:
        x (tf.Tensor): Le tenseur d'entrée pour les images.
        y (tf.Tensor): Le tenseur d'entrée pour les étiquettes.
    Retourne:
        tuple: Un tuple contenant:
            - y_pred (tf.Tensor): Les probabilités prédites pour chaque classe.
            - train_op (tf.Operation): L'opération pour entraîner le modèle.
            - loss (tf.Tensor): Le tenseur de perte.
            - accuracy (tf.Tensor): Le tenseur de précision.
    """
    # Initialisation He avec scale=2.0
    he_init = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Conv1
    conv1 = tf.layers.Conv2D(
        filters=6, kernel_size=5, padding='same',
        activation=tf.nn.relu,
        kernel_initializer=he_init)(x)

    pool1 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

    # Conv2
    conv2 = tf.layers.Conv2D(
        filters=16, kernel_size=5, padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=he_init)(pool1)

    pool2 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)

    # Flatten
    flat = tf.layers.Flatten()(pool2)

    # Dense Layers
    dense1 = tf.layers.Dense(
        120, activation=tf.nn.relu,
        kernel_initializer=he_init)(flat)

    dense2 = tf.layers.Dense(
        84, activation=tf.nn.relu,
        kernel_initializer=he_init)(dense1)

    # Logits (sans activation)
    logits = tf.layers.Dense(10, kernel_initializer=he_init)(dense2)

    # Softmax pour la prédiction
    y_pred = tf.nn.softmax(logits)

    # Calcul de la perte AVEC les logits
    loss = tf.losses.softmax_cross_entropy(y, logits)

    # Optimisation
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Calcul précision
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return y_pred, train_op, loss, accuracy
