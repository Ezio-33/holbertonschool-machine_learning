#!/usr/bin/env python3
"""Module implémentant la classe YOLO v3 pour la détection d'objets"""

import tensorflow.keras as K


class Yolo:
    """Classe pour charger et configurer le modèle YOLO v3

    Attributes:
        model (keras.Model): Modèle Darknet chargé
        class_names (list): Liste des noms de classes COCO
        class_t (float): Seuil de confiance des détections
        nms_t (float): Seuil de suppression non-maximale
        anchors (numpy.ndarray): Boîtes d'ancrage prédéfinies
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialise les paramètres YOLO

        Args:
            model_path (str): Chemin vers le fichier .h5 du modèle
            classes_path (str): Chemin vers le fichier des classes COCO
            class_t (float): Seuil de confiance [0,1]
            nms_t (float): Seuil NMS [0,1]
            anchors (np.ndarray): Boîtes d'ancrage (outputs, nb_ancres, 2)
        """

        # Chargement du modèle Keras
        self.model = K.models.load_model(model_path)

        # Lecture des noms de classes
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        # Initialisation des paramètres
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
