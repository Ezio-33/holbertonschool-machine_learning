#!/usr/bin/env python3
"""Module de filtrage des prédictions YOLO"""

import tensorflow.keras as K
import numpy as np


class Yolo():
	"""
	Classe Yolo qui utilise l'algorithme Yolo v3 pour effectuer la détection d'objets
	"""

	def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
		"""
		Constructeur de la classe

		Arguments:
		 - model_path est le chemin où un modèle Darknet Keras est stocké
		 - classes_path est le chemin où la liste des noms de classes utilisés pour
			le modèle Darknet, listés dans l'ordre des index, peut être trouvée
		 - class_t est un flottant représentant le seuil de score de boîte pour
			l'étape de filtrage initiale
		 - nms_t est un flottant représentant le seuil IOU pour
			la suppression non maximale
		 - anchors est un numpy.ndarray de forme (outputs, anchor_boxes, 2)
			contenant toutes les boîtes d'ancrage :
			* outputs est le nombre de sorties (prédictions) faites par
				le modèle Darknet
			* anchor_boxes est le nombre de boîtes d'ancrage utilisées pour
				chaque prédiction
			* 2 => [largeur_boîte_ancrage, hauteur_boîte_ancrage]

		Attributs d'instance publics :
		 - model : le modèle Darknet Keras
		 - class_names : une liste des noms de classes pour le modèle
		 - class_t : le seuil de score de boîte pour l'étape de filtrage initiale
		 - nms_t : le seuil IOU pour la suppression non maximale
		 - anchors : les boîtes d'ancrage
		"""

		self.model = K.models.load_model(model_path)

		with open(classes_path, 'r') as f:
			self.class_names = [line.strip() for line in f]

		self.class_t = class_t
		self.nms_t = nms_t
		self.anchors = anchors

	def sigmoid(self, x):
		"""
		Fonction qui calcule la sigmoïde
		"""
		return 1 / (1 + np.exp(-x))

	# Méthode publique
	def process_outputs(self, outputs, image_size):
		"""
		Méthode publique pour traiter les sorties

		Arguments:
		 - outputs est une liste de numpy.ndarrays contenant les prédictions
			du modèle Darknet pour une seule image :
			Chaque sortie aura la forme
			(grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
			 * grid_height & grid_width => la hauteur et la largeur de
				la grille utilisée pour la sortie
			 * anchor_boxes => le nombre de boîtes d'ancrage utilisées
			 * 4 => (t_x, t_y, t_w, t_h)
			 * 1 => confiance_boîte
			 * classes => probabilités de classe pour toutes les classes
		 - image_size est un numpy.ndarray contenant la taille originale de l'image
			[image_size[0], image_size[1]]

		Renvoie :
		 Un tuple de (boxes, box_confidences, box_class_probs) :
		 - boxes : une liste de numpy.ndarrays de forme
				(grid_height, grid_width, anchor_boxes, 4)
			contenant les boîtes de délimitation traitées pour chaque sortie :
			* 4 => (x1, y1, x2, y2)
			* (x1, y1, x2, y2) devraient représenter la boîte de délimitation
				relative à l'image originale
		 - box_confidences : une liste de numpy.ndarrays de forme
			(grid_height, grid_width, anchor_boxes, 1)
			contenant les confiances des boîtes pour chaque sortie, respectivement
		 - box_class_probs : une liste de numpy.ndarrays de forme
			(grid_height, grid_width, anchor_boxes, classes)
			contenant les probabilités de classe des boîtes
			pour chaque sortie, respectivement
		"""

		img_height = image_size[0]
		img_width = image_size[1]

		boxes = []
		box_confidences = []
		box_class_probs = []
		for output in outputs:
			# Créer la liste avec np.ndarray
			boxes.append(output[..., 0:4])
			# Calculer les confiances pour chaque sortie
			box_confidences.append(self.sigmoid(output[..., 4, np.newaxis]))
			# Calculer la probabilité de classe pour chaque sortie
			box_class_probs.append(self.sigmoid(output[..., 5:]))

		for i, box in enumerate(boxes):
			grid_height, grid_width, anchor_boxes, _ = box.shape

			c = np.zeros((grid_height, grid_width, anchor_boxes), dtype=int)

			# Matrice Cy
			idx_y = np.arange(grid_height)
			idx_y = idx_y.reshape(grid_height, 1, 1)
			Cy = c + idx_y

			# Matrice Cx
			idx_x = np.arange(grid_width)
			idx_x = idx_x.reshape(1, grid_width, 1)
			Cx = c + idx_x

			# Coordonnées centrales de sortie et normalisées
			tx = (box[..., 0])
			ty = (box[..., 1])
			tx_n = self.sigmoid(tx)
			ty_n = self.sigmoid(ty)

			# Calculer bx & by et les normaliser
			bx = tx_n + Cx
			by = ty_n + Cy
			bx /= grid_width
			by /= grid_height

			# Calculer tw & th
			tw = (box[..., 2])
			th = (box[..., 3])
			tw_t = np.exp(tw)
			th_t = np.exp(th)

			# Dimensions des boîtes d'ancrage
			pw = self.anchors[i, :, 0]
			ph = self.anchors[i, :, 1]

			# Calculer bw & bh et les normaliser
			bw = pw * tw_t
			bh = ph * th_t
			# taille d'entrée
			input_width = self.model.input.shape[1]
			input_height = self.model.input.shape[2]
			bw /= input_width
			bh /= input_height

			# Coordonnées des coins
			x1 = bx - bw / 2
			y1 = by - bh / 2
			x2 = x1 + bw
			y2 = y1 + bh

			# Ajuster l'échelle
			box[..., 0] = x1 * img_width
			box[..., 1] = y1 * img_height
			box[..., 2] = x2 * img_width
			box[..., 3] = y2 * img_height

		return boxes, box_confidences, box_class_probs

	# Méthode publique
	def filter_boxes(self, boxes, box_confidences, box_class_probs):
		"""
		Méthode publique pour filtrer les boîtes

		Arguments:
		 - boxes : une liste de numpy.ndarrays de forme
			 (grid_height, grid_width, anchor_boxes, 4)
			contenant les boîtes de délimitation traitées pour chaque sortie
		 - box_confidences : une liste de numpy.ndarrays de forme
			 (grid_height, grid_width, anchor_boxes, 1)
			contenant les confiances des boîtes traitées pour chaque sortie
		 - box_class_probs : une liste de numpy.ndarrays de forme
			 (grid_height, grid_width, anchor_boxes, classes)
			contenant les probabilités de classe des boîtes traitées pour chaque sortie
		Renvoie :
		 Un tuple de (filtered_boxes, box_classes, box_scores) :
		 * filtered_boxes : un numpy.ndarray de forme (?, 4) contenant
			toutes les boîtes de délimitation filtrées :
		 * box_classes : un numpy.ndarray de forme (?,) contenant
			le numéro de classe que chaque boîte dans filtered_boxes prédit,
			respectivement
		 * box_scores : un numpy.ndarray de forme (?) contenant
			les scores des boîtes pour chaque boîte dans filtered_boxes, respectivement
		"""

		scores = []

		for box_conf, box_class_prob in zip(box_confidences, box_class_probs):
			scores.append(box_conf * box_class_prob)

		# scores des boîtes
		box_scores = [score.max(axis=-1) for score in scores]
		box_scores = [box.reshape(-1) for box in box_scores]
		box_scores = np.concatenate(box_scores)
		filtering_mask = np.where(box_scores < self.class_t)
		box_scores = np.delete(box_scores, filtering_mask)

		# classes des boîtes
		box_classes = [box.argmax(axis=-1) for box in scores]
		box_classes = [box.reshape(-1) for box in box_classes]
		box_classes = np.concatenate(box_classes)
		box_classes = np.delete(box_classes, filtering_mask)

		# boîtes filtrées
		filtered_boxes_list = [box.reshape(-1, 4) for box in boxes]
		filtered_boxes_box = np.concatenate(filtered_boxes_list, axis=0)
		filtered_boxes = np.delete(filtered_boxes_box, filtering_mask, axis=0)

		return (filtered_boxes, box_classes, box_scores)
