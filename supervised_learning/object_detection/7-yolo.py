#!/usr/bin/env python3
"""Module pour charger des images depuis un dossier"""

import cv2
import glob
import numpy as np
import tensorflow.keras as K
import os


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
        Filtre les boîtes selon les scores de classe et calcule
        les scores finaux

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

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applique la suppression non-maximale pour filtrer les boîtes
        superposées

        Args:
            filtered_boxes (np.ndarray): Boîtes filtrées shape (N,4)
            box_classes (np.ndarray): Classes des boîtes shape (N,)
            box_scores (np.ndarray): Scores des boîtes shape (N,)

        Returns:
            tuple: (boîtes_finales, classes_finales, scores_finals)
        """
        selected_indices = []

        # Parcours chaque classe unique triée
        for cls in np.unique(box_classes):
            # Masque pour la classe courante
            cls_mask = box_classes == cls
            cls_indices = np.where(cls_mask)[0]

            if len(cls_indices) == 0:
                continue

            # Tri par score décroissant
            sorted_scores_idx = np.argsort(-box_scores[cls_indices])
            sorted_indices = cls_indices[sorted_scores_idx]

            while len(sorted_indices) > 0:
                # Sélectionne la boîte avec le score le plus haut
                current_idx = sorted_indices[0]
                selected_indices.append(current_idx)

                if len(sorted_indices) == 1:
                    break

                # Calcul IoU avec les autres boîtes
                current_box = filtered_boxes[current_idx]
                other_boxes = filtered_boxes[sorted_indices[1:]]

                # Calcul des coordonnées d'intersection
                x1 = np.maximum(current_box[0], other_boxes[:, 0])
                y1 = np.maximum(current_box[1], other_boxes[:, 1])
                x2 = np.minimum(current_box[2], other_boxes[:, 2])
                y2 = np.minimum(current_box[3], other_boxes[:, 3])

                intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
                area_current = (
                    (current_box[2] - current_box[0]) *
                    (current_box[3] - current_box[1])
                )
                area_others = (
                    (other_boxes[:, 2] - other_boxes[:, 0]) *
                    (other_boxes[:, 3] - other_boxes[:, 1])
                )
                union = area_current + area_others - intersection

                # Évite la division par zéro
                iou = intersection / (union + 1e-6)

                # Filtrage des boîtes trop proches
                keep_mask = iou <= self.nms_t
                sorted_indices = sorted_indices[1:][keep_mask]

        if not selected_indices:
            return np.array([]), np.array([]), np.array([])

        # Extraction des résultats
        selected_indices = np.array(selected_indices)
        return (
            filtered_boxes[selected_indices],
            box_classes[selected_indices],
            box_scores[selected_indices]
        )

    @staticmethod
    def load_images(folder_path):
        """
        Charge toutes les images d'un dossier donné.

        Args:
            folder_path (str): Chemin vers le dossier contenant les images.

        Returns:
            tuple: (images, image_paths)
                - images: Liste de numpy.ndarray représentant
                les images chargées.
                - image_paths: Liste des chemins complets vers chaque image.
        """
        # Initialisation des listes
        images = []
        image_paths = []

        # Parcours des fichiers dans le dossier
        for file_path in glob.glob(f"{folder_path}/*"):
            # Charger l'image avec OpenCV
            image = cv2.imread(file_path)
            if image is not None:  # Vérifier que l'image est valide
                images.append(image)
                image_paths.append(file_path)

        # Vérifier si aucune image n'a été trouvée
        if len(images) == 0:
            raise ValueError(
                f"Aucune image valide trouvée dans le dossier : {folder_path}"
            )

        return images, image_paths

    def preprocess_images(self, images):
        """
        Prétraite une liste d'images pour le modèle YOLO.

        Args:
            images (list): Liste d'images sous forme de numpy.ndarray.

        Returns:
            tuple: (pimages, image_shapes)
                - pimages: numpy.ndarray contenant toutes
                les images prétraitées.
                - image_shapes: numpy.ndarray contenant les
                dimensions originales de chaque image sous
                forme (hauteur, largeur).
        """
        # Récupérer la taille d'entrée du modèle
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        # Initialisation des listes pour stocker les résultats
        lpimages = []
        limage_shapes = []

        for img in images:
            # Sauvegarder la taille originale de l'image
            img_shape = img.shape[0], img.shape[1]
            limage_shapes.append(img_shape)

            # Redimensionner l'image à la taille du modèle
            dimension = (input_w, input_h)
            resized = cv2.resize(img, dimension, interpolation=cv2.INTER_CUBIC)

            # Normaliser les pixels entre 0 et 1
            pimage = resized / 255.0

            # Ajouter l'image prétraitée à la liste
            lpimages.append(pimage)

        # Convertir en tableaux NumPy
        pimages = np.array(lpimages)
        image_shapes = np.array(limage_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Affiche une image avec toutes ses boîtes englobantes, noms de classes
        et scores associés.

        Args:
            image (numpy.ndarray): Image non traitée.
            boxes (numpy.ndarray): Coordonnées des boîtes englobantes.
            box_classes (numpy.ndarray): Indices des classes pour chaque boîte.
            box_scores (numpy.ndarray): Scores associés à chaque boîte.
            file_name (str): Nom du fichier original de l'image.
        """
        # Arrondir les scores à 2 décimales
        box_scores_r = np.around(box_scores, decimals=2)

        # Parcourir toutes les boîtes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            class_name = self.class_names[box_classes[i]]
            score = str(box_scores_r[i])

            # Dessiner le rectangle autour de la boîte
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # Ajouter le texte au-dessus de la boîte
            text = f"{class_name} {score}"
            cv2.putText(image, text, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Afficher l'image dans une fenêtre
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        # Sauvegarder ou fermer selon la touche pressée
        if key == ord('s'):
            if not os.path.exists('./detections'):
                os.makedirs('./detections')
            cv2.imwrite(f"./detections/{file_name}", image)
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Effectue des prédictions sur toutes les images d'un dossier.
        Args:
            folder_path: Chemin vers le dossier contenant les images.
        Returns:
            Tuple (predictions, image_paths) avec les résultats.
        """
        predictions = []
        images, image_paths = self.load_images(folder_path)

        if not images:
            return [], []

        # Prétraitement des images
        pimages, image_shapes = self.preprocess_images(images)

        # Prédiction du modèle (sortie shape [batch, grid_h, grid_w, ancres, 85])
        model_outputs = self.model.predict(pimages)

        for i, img in enumerate(images):
            # Extraction CORRECTE des sorties pour chaque image
            outputs = [
                model_outputs[0][i, :, :, :, :],  # Couche 13x13
                model_outputs[1][i, :, :, :, :],  # Couche 26x26
                model_outputs[2][i, :, :, :, :]   # Couche 52x52
            ]

            # Traitement des sorties
            boxes, box_conf, box_probs = self.process_outputs(outputs, image_shapes[i])

            # Filtrage et NMS
            filtered_boxes, box_classes, box_scores = self.filter_boxes(boxes, box_conf, box_probs)
            final_boxes, final_classes, final_scores = self.non_max_suppression(filtered_boxes, box_classes, box_scores)

            # Affichage et stockage
            base_name = os.path.basename(image_paths[i])
            self.show_boxes(img.copy(), final_boxes, final_classes, final_scores, base_name)
            predictions.append((final_boxes, final_classes, final_scores))

        return predictions, image_paths
