#!/usr/bin/env python3
"""Tâche 9"""
import numpy as np
from scipy import stats
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest:
	"""
	La classe Random_Forest implémente un algorithme de forêt aléatoire
	qui construit une grande liste d'arbres de décision avec des critères
	de division aléatoires.

	Attributs:
	n_trees : int
		Nombre d'arbres dans la forêt.
	max_depth : int
		Profondeur maximale des arbres.
	min_pop : int
		Population minimale à un nœud pour qu'il se divise.
	seed : int
		Graine pour la génération de nombres aléatoires.
	numpy_preds : list
		Liste des fonctions de prédiction de chaque arbre.
	target : array-like
		Variable cible utilisée pendant l'entraînement.
	explanatory : array-like
		Variables explicatives utilisées pendant l'entraînement.

	Méthodes:
	__init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
		Initialise la forêt aléatoire avec les paramètres spécifiés.

	predict(self, explanatory):
		Prédit les étiquettes de classe pour les données explicatives données
		en fonction du vote majoritaire de tous les arbres.

	fit(self, explanatory, target, n_trees=100, verbose=0):
		Entraîne la forêt aléatoire sur les données explicatives et cibles
		données en construisant des arbres de décision.

	accuracy(self, test_explanatory, test_target):
		Calcule la précision de la forêt aléatoire sur les données de test.
	"""

	def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
		"""
		Initialise la forêt aléatoire avec les paramètres spécifiés.

		Paramètres:
		n_trees : int, optionnel
			Nombre d'arbres dans la forêt (par défaut 100).
		max_depth : int, optionnel
			Profondeur maximale des arbres (par défaut 10).
		min_pop : int, optionnel
			Population minimale à un nœud pour qu'il se divise (par défaut 1).
		seed : int, optionnel
			Graine pour la génération de nombres aléatoires (par défaut 0).
		"""
		self.numpy_predicts = []
		self.target = None
		self.numpy_preds = None
		self.n_trees = n_trees
		self.max_depth = max_depth
		self.min_pop = min_pop
		self.seed = seed

	def predict(self, explanatory):
		"""
		Prédit les étiquettes de classe pour les données explicatives données.

		Paramètres:
		explanatory : array-like
			Variables explicatives pour lesquelles des prédictions sont requises.

		Retourne:
		array-like
			Étiquettes de classe prédites.
		"""
		all_preds = []
		for tree_predict in self.numpy_preds:
			preds = tree_predict(explanatory)
			all_preds.append(preds)
		all_preds = np.array(all_preds)
		mode_preds = stats.mode(all_preds, axis=0)[0]
		return mode_preds.flatten()

	def fit(self, explanatory, target, n_trees=100, verbose=0):
		"""
		Entraîne la forêt aléatoire sur les données explicatives et cibles données.

		Paramètres:
		explanatory : array-like
			Variables explicatives utilisées pour l'entraînement.
		target : array-like
			Variable cible utilisée pour l'entraînement.
		n_trees : int, optionnel
			Nombre d'arbres dans la forêt (par défaut 100).
		verbose : int, optionnel
			Si défini à 1, affiche les statistiques d'entraînement (par défaut 0).
		"""
		self.target = target
		self.explanatory = explanatory
		self.numpy_preds = []
		depths = []
		nodes = []
		leaves = []
		accuracies = []
		for i in range(n_trees):
			T = Decision_Tree(max_depth=self.max_depth,
							  min_pop=self.min_pop, seed=self.seed + i)
			T.fit(explanatory, target)
			self.numpy_preds.append(T.predict)
			depths.append(T.depth())
			nodes.append(T.count_nodes())
			leaves.append(T.count_nodes(only_leaves=True))
			accuracies.append(T.accuracy(T.explanatory, T.target))
		if verbose == 1:
			print(f"""  Entraînement terminé.
	- Profondeur moyenne               : {np.array(depths).mean()}
	- Nombre moyen de nœuds           : {np.array(nodes).mean()}
	- Nombre moyen de feuilles         : {np.array(leaves).mean()}
	- Précision moyenne sur les données d'entraînement : {np.array(accuracies).mean()}""")
			print(f"    - Précision de la forêt sur td   : "
				  f"{self.accuracy(self.explanatory, self.target)}")

	def accuracy(self, test_explanatory, test_target):
		"""
		Calcule la précision de la forêt aléatoire sur les données de test.

		La précision est calculée comme la proportion d'étiquettes correctement prédites
		sur le total des étiquettes.

		Paramètres:
		test_explanatory : array-like
			Variables explicatives des données de test.
		test_target : array-like
			Étiquettes cibles réelles des données de test.

		Retourne:
		float
			Précision de la forêt aléatoire sur les données de test.
		"""

		return np.sum(np.equal(self.predict(test_explanatory),
							   test_target)) / test_target.size
