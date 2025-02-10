#!/usr/bin/env python3
"""Architecture LeNet-5 avec TensorFlow 1.x

Ce module construit une version modifiée de l’architecture LeNet-5 en
utilisant TensorFlow 1.x.
Il retourne le tenseur de prédiction softmax, l’opération
d’entraînement (Adam), le tenseur de perte et le tenseur de précision.
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def lenet5(x, y):
    """
    Construit l’architecture LeNet-5 modifiée.

    Args:
        x: Placeholder de forme (m, 28, 28, 1) contenant les images d'entrée.
        y: Placeholder de forme (m, 10) contenant les labels one-hot.

    Returns:
        y_out: Tenseur de sortie activé par softmax (prédictions).
        train: Opération d’entraînement utilisant l’optimisateur Adam.
        loss: Tenseur de la perte.
        acc: Tenseur de la précision.
    """
    # Initialiseur He avec réplicabilité (version TF1 via tf.contrib)
    k_init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu

    # Couche 1 : convolution avec 6 filtres 5x5 avec padding "same"
    layer_1 = tf.layers.conv2d(
        inputs=x,
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=activation,
        kernel_initializer=k_init
    )
    # Exemple : Si x est de forme (m, 28, 28, 1), alors layer_1 aura forme (m,
    # 28, 28, 6).

    # Couche 2 : max pooling avec un kernel 2x2 et stride 2
    pool_1 = tf.layers.max_pooling2d(
        inputs=layer_1,
        pool_size=(2, 2),
        strides=(2, 2)
    )
    # Exemple : (m, 28, 28, 6) devient (m, 14, 14, 6).

    # Couche 3 : convolution avec 16 filtres 5x5 avec padding "valid"
    layer_2 = tf.layers.conv2d(
        inputs=pool_1,
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=activation,
        kernel_initializer=k_init
    )
    # Avec valid padding, si l’entrée est 14x14, la sortie sera de dimension
    # (14-5+1)=10, soit (m, 10, 10, 16).

    # Couche 4 : max pooling avec kernel 2x2 et stride 2
    pool_2 = tf.layers.max_pooling2d(
        inputs=layer_2,
        pool_size=(2, 2),
        strides=(2, 2)
    )
    # La sortie passera de (m, 10, 10, 16) à (m, 5, 5, 16).

    # Aplatissement des sorties pour les couches entièrement connectées
    flat = tf.layers.flatten(pool_2)
    # Chaque échantillon devient un vecteur de 5*5*16 = 400 éléments.

    # Couche 5 : couche dense (fully connected) avec 120 neurones
    dense1 = tf.layers.dense(
        inputs=flat,
        units=120,
        activation=activation,
        kernel_initializer=k_init
    )

    # Couche 6 : couche dense avec 84 neurones
    dense2 = tf.layers.dense(
        inputs=dense1,
        units=84,
        activation=activation,
        kernel_initializer=k_init
    )

    # Couche 7 : couche de sortie dense avec 10 neurones (sans activation)
    # On ne lui applique pas d’activation ici car la fonction de perte attend
    # des logits (sortie non activée)
    output_layer = tf.layers.dense(
        inputs=dense2,
        units=10,
        kernel_initializer=k_init
    )

    # Prédictions : application de softmax sur les logits pour obtenir des
    # probabilités
    y_out = tf.nn.softmax(output_layer)

    # Calcul de la perte avec softmax_cross_entropy, qui applique softmax de
    # façon interne sur les logits
    loss = tf.losses.softmax_cross_entropy(y, output_layer)

    # Opération d’entraînement avec l’optimiseur Adam (hyperparamètres par
    # défaut)
    train = tf.train.AdamOptimizer().minimize(loss)

    # Calcul de la précision : compare l’indice de la classe prédite (argmax
    # des logits) aux labels réels
    equality = tf.equal(tf.argmax(y, 1), tf.argmax(output_layer, 1))
    acc = tf.reduce_mean(tf.cast(equality, tf.float32))

    return y_out, train, loss, acc
