#!/usr/bin/env python3
"""
Module pour évaluer la sortie d'un réseau neuronal
"""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    Évalue la sortie d'un réseau neuronal

    Args:
        X: numpy.ndarray contenant les données d'entrée à évaluer
        Y: numpy.ndarray contenant les étiquettes one-hot
        save_path: emplacement du modèle à charger

    Returns:
        prédiction du réseau, précision et perte respectivement
    """
    # Création d'une nouvelle session
    with tf.Session() as sess:
        # Importation du métagraphe sauvegardé
        saver = tf.train.import_meta_graph(save_path + '.meta')
        # Restauration des variables du modèle
        saver.restore(sess, save_path)

        # Récupération des tenseurs nécessaires
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        # Évaluation du modèle
        feed_dict = {x: X, y: Y}
        prediction, model_accuracy, model_cost = sess.run(
            [y_pred, accuracy, loss],
            feed_dict=feed_dict
        )

        return prediction, model_accuracy, model_cost
