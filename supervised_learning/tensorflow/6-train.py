#!/usr/bin/env python3
"""
Module pour construire, entraîner et sauvegarder un réseau neuronal
"""
import tensorflow.compat.v1 as tf


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
    Construit, entraîne et sauvegarde un réseau neuronal classifieur

    Args:
        X_train: données d'entraînement
        Y_train: étiquettes d'entraînement
        X_valid: données de validation
        Y_valid: étiquettes de validation
        layer_sizes: tailles des couches
        activations: fonctions d'activation
        alpha: taux d'apprentissage
        iterations: nombre d'itérations
        save_path: chemin de sauvegarde

    Returns:
        chemin où le modèle a été sauvegardé
    """
    # Import des fonctions nécessaires
    create_placeholders = __import__(
        '0-create_placeholders').create_placeholders
    forward_prop = __import__('2-forward_prop').forward_prop
    calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    create_train_op = __import__('5-create_train_op').create_train_op

    # Création du graphe
    x, y = create_placeholders(X_train.shape[1], layer_sizes[-1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    # Ajout au graphe
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Initialisation et création du saver
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Création de la session et entraînement
    with tf.Session() as sess:
        sess.run(init)

        # Boucle d'entraînement
        for i in range(iterations + 1):
            # Calcul des métriques d'entraînement
            train_cost, train_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train}
            )

            # Calcul des métriques de validation
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid}
            )

            # Affichage des métriques
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))

            if i < iterations:
                # Exécution d'une étape d'entraînement
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        # Sauvegarde du modèle
        save_path = saver.save(sess, save_path)

    return save_path
