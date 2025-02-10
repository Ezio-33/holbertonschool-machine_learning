#!/usr/bin/env python3
"""LeNet-5 avec TensorFlow 1.x"""
import tensorflow as tf  # Import de TensorFlow 1.x


def lenet5(x, y):
    # 1. Initialisation "He Normal" (compatible TF1)
    k_init = tf.contrib.layers.variance_scaling_initializer()

    # 2. Couche Conv1 : 6 filtres 5x5, padding 'same' pour conserver la taille
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=k_init)(x)

    # 3. Pooling1 : Réduction 2x2 avec pas de 2
    pool1 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)

    # 4. Couche Conv2 : 16 filtres 5x5, padding 'valid' (pas de padding)
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=k_init)(pool1)

    # 5. Pooling2 : Même paramètres
    pool2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)

    # 6. Mise à plat pour les couches denses
    flat = tf.layers.Flatten()(pool2)

    # 7. Couche Dense1 : 120 neurones avec ReLU
    fc1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                          kernel_initializer=k_init)(flat)

    # 8. Couche Dense2 : 84 neurones avec ReLU
    fc2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                          kernel_initializer=k_init)(fc1)

    # 9. Couche de Sortie : 10 neurones (sans activation)
    logits = tf.layers.Dense(units=10, kernel_initializer=k_init)(fc2)

    # 10. Application du Softmax pour obtenir les probabilités
    y_pred = tf.nn.softmax(logits)

    # 11. Calcul de la perte (cross-entropy) sur les logits directement
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y, logits))

    # 12. Optimiseur Adam
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # 13. Calcul de la précision
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return y_pred, train_op, loss, acc
