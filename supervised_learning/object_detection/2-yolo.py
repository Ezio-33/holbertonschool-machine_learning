#!/usr/bin/env python3
"""Module de filtrage des prédictions YOLO"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """Classe YOLO v3 complète avec correction d'erreur"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Fonction sigmoïde vectorisée"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Traitement des sorties du réseau avec correction de l'erreur .value
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        for output_idx, output in enumerate(outputs):
            grid_h, grid_w, nb_ancres, _ = output.shape

            # Initialisation des tableaux de sortie
            boxes_layer = np.zeros((grid_h, grid_w, nb_ancres, 4))

            for i in range(grid_h):
                for j in range(grid_w):
                    for k in range(nb_ancres):
                        tx, ty, tw, th = output[i, j, k, :4]

                        # Calcul des coordonnées normalisées
                        cx = (j + self.sigmoid(tx)) / grid_w
                        cy = (i + self.sigmoid(ty)) / grid_h

                        # Récupération des ancres correspondantes
                        pw, ph = self.anchors[output_idx][k]

                        # Calcul des dimensions de la boîte
                        bw = pw * np.exp(tw) / input_width
                        bh = ph * np.exp(th) / input_height

                        # Conversion finale en coordonnées image
                        x1 = (cx - bw / 2) * image_size[1]
                        y1 = (cy - bh / 2) * image_size[0]
                        x2 = (cx + bw / 2) * image_size[1]
                        y2 = (cy + bh / 2) * image_size[0]

                        boxes_layer[i, j, k] = [x1, y1, x2, y2]

            # Stockage des résultats pour cette couche
            boxes.append(boxes_layer)
            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filtre les boîtes selon les scores de classe et calcule les
        scores finaux

        Args:
            boxes: Liste de tableaux numpy des coordonnées des boîtes
            box_confidences: Liste de tableaux numpy des confiances
            box_class_probs: Liste de tableaux numpy des probabilités de classe

        Returns:
            Tuple (filtered_boxes, filtered_classes, filtered_scores)
        """
        filtered_boxes = []
        filtered_classes = []
        filtered_scores = []

        # Pour chaque couche de sortie (13x13, 26x26, 52x52)
        for box_layer, conf_layer, prob_layer in zip(
                boxes, box_confidences, box_class_probs):
            # Aplatir les tableaux pour simplifier le traitement
            box_flat = box_layer.reshape(-1, 4)
            conf_flat = conf_layer.reshape(-1)
            prob_flat = prob_layer.reshape(-1, len(self.class_names))

            # Étape 1 : Trouver la classe dominante et son score
            class_indices = np.argmax(prob_flat, axis=1)
            class_scores = np.max(prob_flat, axis=1)
            final_scores = conf_flat * class_scores

            # Étape 2 : Filtrer par seuil de confiance
            mask = final_scores >= self.class_t
            filtered_boxes.extend(box_flat[mask])
            filtered_classes.extend(class_indices[mask])
            filtered_scores.extend(final_scores[mask])

        return (
            np.array(filtered_boxes),
            np.array(filtered_classes),
            np.array(filtered_scores)
        )
