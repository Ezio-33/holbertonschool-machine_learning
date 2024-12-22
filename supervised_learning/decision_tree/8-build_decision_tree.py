#!/usr/bin/env python3

"""
Ceci est le module 1-build_decision_tree.
"""

import numpy as np


class Node:
        """
        Représente un nœud dans un arbre de décision.

        Attributs:
                feature (int): L'indice de la caractéristique utilisée pour la division
                        à ce nœud.
                threshold (float): La valeur seuil utilisée pour la division à ce nœud.
                left_child (Node): Le nœud enfant gauche.
                right_child (Node): Le nœud enfant droit.
                is_leaf (bool): Indique si ce nœud est une feuille.
                is_root (bool): Indique si ce nœud est la racine.
                sub_population (None ou ndarray): Le sous-ensemble de la population
                        atteignant ce nœud.
                depth (int): La profondeur de ce nœud dans l'arbre de décision.
        """

        def __init__(self, feature=None, threshold=None, left_child=None,
                                 right_child=None, is_root=False, depth=0):
                self.feature = feature
                self.threshold = threshold
                self.left_child = left_child
                self.right_child = right_child
                self.is_leaf = False
                self.is_root = is_root
                self.sub_population = None
                self.depth = depth

        def max_depth_below(self):
                """
                Calcule récursivement la profondeur maximale de la sous-arborescence
                en dessous de ce nœud.

                Retourne:
                        int: La profondeur maximale en dessous de ce nœud.
                """
                if self.is_leaf:
                        return self.depth

                return max(self.left_child.max_depth_below(),
                                   self.right_child.max_depth_below())

        def count_nodes_below(self, only_leaves=False):
                """
                Retourne le nombre de nœuds sous ce nœud.
                Si only_leaves est True, ne compte que les feuilles.
                """
                if only_leaves and self.is_leaf:
                        return 1

                if not self.is_leaf:
                        # NOTE Compte le nœud actuel uniquement si only_leaves == False
                        return self.left_child.count_nodes_below(only_leaves=only_leaves)\
                                + self.right_child.count_nodes_below(only_leaves=only_leaves)\
                                + (not only_leaves)

        def __str__(self):
                """
                Affiche la représentation en chaîne de caractères du nœud et de ses enfants.
                """

                if self.is_root:
                        s = "racine"
                else:
                        s = "-> nœud"

                return f"{s} [feature={self.feature}, threshold={self.threshold}]\n"\
                        + self.left_child_add_prefix(str(self.left_child))\
                        + self.right_child_add_prefix(str(self.right_child))

        def left_child_add_prefix(self, text):
                """
                Ajoute la représentation en chaîne de caractères de l'enfant gauche
                au texte donné.
                """
                lines = text.split("\n")
                new_text = "    +--" + lines[0] + "\n"
                for x in lines[1:]:
                        new_text += ("    |  " + x) + "\n"
                return (new_text)

        def right_child_add_prefix(self, text):
                """
                Ajoute la représentation en chaîne de caractères de l'enfant droit
                au texte donné.
                """
                lines = text.split("\n")
                new_text = "    +--" + lines[0] + "\n"
                for x in lines[1:]:
                        new_text += ("       " + x) + "\n"
                # NOTE A dû enlever la nouvelle ligne supplémentaire après le nœud droit
                # Il pourrait y avoir de meilleures alternatives
                return (new_text.rstrip())

        def get_leaves_below(self):
                """
                Retourne la liste de toutes les feuilles en dessous de celle-ci.
                """

                return self.left_child.get_leaves_below()\
                        + self.right_child.get_leaves_below()

        def update_bounds_below(self):
                """
                Calcule récursivement, pour chaque nœud, deux dictionnaires stockés en
                tant qu'attributs Node.lower et Node.upper.
                """
                if self.is_root:
                        self.lower = {0: -1 * np.inf}
                        self.upper = {0: np.inf}

                for child in [self.left_child, self.right_child]:
                        child.upper = self.upper.copy()
                        child.lower = self.lower.copy()

                if self.feature in self.left_child.lower.keys():
                        # Mise à jour avec le seuil inférieur gauche
                        self.left_child.lower[self.feature] = \
                                max(self.threshold, self.left_child.lower[self.feature])
                else:
                        self.left_child.lower[self.feature] = self.threshold

                if self.feature in self.right_child.upper.keys():
                        # Mise à jour avec le seuil supérieur droit
                        self.right_child.upper[self.feature] = \
                                min(self.threshold, self.right_child.upper[self.feature])
                else:
                        self.right_child.upper[self.feature] = self.threshold

                self.left_child.update_bounds_below()
                self.right_child.update_bounds_below()

        def update_indicator(self):
                """
                Met à jour la fonction indicatrice basée sur les bornes inférieures et
                supérieures.

                La fonction indicatrice est une fonction lambda qui prend un tableau 2D
                numpy `x` représentant les caractéristiques des individus et retourne un
                tableau 1D numpy de taille `n_individuals` où le `i`-ème élément est
                `True` si le `i`-ème individu satisfait les conditions spécifiées
                par les bornes inférieure et supérieure.
                """
                def is_large_enough(x):
                        return np.all(
                                np.array([np.greater(x[:, key], self.lower[key])
                                                  for key in self.lower]), axis=0
                        )

                def is_small_enough(x):
                        return np.all(
                                np.array([np.less_equal(x[:, key], self.upper[key])
                                                  for key in self.upper]), axis=0
                        )

                self.indicator = lambda x: np.all(
                        np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

        def pred(self, x):
                """
                Prédit l'étiquette de classe pour un échantillon d'entrée donné.

                Args:
                        x (list): L'échantillon d'entrée pour prédire l'étiquette de classe.

                Returns:
                        int: L'étiquette de classe prédite pour l'échantillon d'entrée
                                (valeur de la feuille).
                """
                if x[self.feature] > self.threshold:
                        return self.left_child.pred(x)
                else:
                        return self.right_child.pred(x)


class Leaf(Node):
        """
        Représente un nœud feuille dans un arbre de décision.

        Attributs:
                value (any): La valeur associée au nœud feuille.
                is_leaf (bool): Indique si le nœud est une feuille.
                depth (int): La profondeur du nœud feuille dans l'arbre de décision.
        """

        def __init__(self, value, depth=None):
                super().__init__()
                self.value = value
                self.is_leaf = True
                self.depth = depth

        def max_depth_below(self):
                """
                Retourne la profondeur maximale en dessous du nœud feuille.

                Retourne:
                        int: La profondeur maximale en dessous du nœud feuille.
                """
                return self.depth

        def count_nodes_below(self, only_leaves=False):
                """
                Remplace la même méthode de la classe Node.
                Retourne 1.
                """
                return 1

        def __str__(self):
                # NOTE a dû ajouter cet espace vide par erreur
                # dépôt checker avec l'erreur : malekmrabti213
                return (f"-> feuille [value={self.value}] ")

        def get_leaves_below(self):
                """
                Retourne cette feuille comme élément de liste.
                """
                return [self]

        def update_bounds_below(self):
                """
                Ne fait rien ?
                """
                pass

        def pred(self, x):
                """
                Retourne la valeur de la feuille.
                """
                return self.value


class Decision_Tree():
        """
        Une classe représentant un arbre de décision.

        Attributs:
                max_depth (int): La profondeur maximale de l'arbre de décision.
                min_pop (int): La population minimale requise pour diviser un nœud.
                seed (int): La valeur de graine pour la génération de nombres aléatoires.
                split_criterion (str): Le critère utilisé pour diviser les nœuds.
                root (Node): Le nœud racine de l'arbre de décision.
                explanatory: La ou les variables explicatives utilisées pour la prédiction.
                target: La variable cible utilisée pour la prédiction.
                predict: La fonction de prédiction utilisée pour faire des prédictions.

        Méthodes:
                depth(): Retourne la profondeur maximale de l'arbre de décision.
        """

        def __init__(self, max_depth=10, min_pop=1, seed=0,
                                 split_criterion="random", root=None):
                self.rng = np.random.default_rng(seed)
                if root:
                        self.root = root
                else:
                        self.root = Node(is_root=True)
                self.explanatory = None
                self.target = None
                self.max_depth = max_depth
                self.min_pop = min_pop
                self.split_criterion = split_criterion
                self.predict = None

        def depth(self):
                """
                Retourne la profondeur maximale de l'arbre de décision.

                Retourne:
                        int: La profondeur maximale de l'arbre de décision.
                """
                return self.root.max_depth_below()

        def count_nodes(self, only_leaves=False):
                """
                Retourne le nombre de nœuds dans l'arbre de décision.
                Si only_leaves est True, ne compte que les feuilles.
                """
                return self.root.count_nodes_below(only_leaves=only_leaves)

        def __str__(self):
                # NOTE plus propre de mettre à jour ceci que d'utiliser la "solution" que j'ai vue
                return f"{self.root.__str__()}\n"

        def get_leaves(self):
                """
                Obtient la liste des feuilles dans l'arbre.
                """
                return self.root.get_leaves_below()

        def update_bounds(self):
                """
                Appelle update_bounds_below().
                """
                self.root.update_bounds_below()

        def pred(self, x):
                """
                Prédit l'étiquette de classe pour un échantillon d'entrée donné.
                Commence la récursion depuis la racine.

                Args:
                        x (array-like): L'échantillon d'entrée à classifier.

                Returns:
                        L'étiquette de classe prédite pour l'échantillon d'entrée.
                """
                return self.root.pred(x)

        def update_predict(self):
                """
                Met à jour la fonction de prédiction de l'arbre de décision.

                Cette méthode met à jour la fonction de prédiction de l'arbre de décision
                en mettant à jour les indicateurs de toutes les feuilles et en créant une
                nouvelle fonction de prédiction basée sur les indicateurs mis à jour.
                Résulte en un tableau de prédictions pour
                """
                self.update_bounds()
                leaves = self.get_leaves()
                for leaf in leaves:
                        leaf.update_indicator()

                # Configuration de self.predict comme une fonction, avec le tableau A en entrée
                self.predict = lambda A: np.array([self.root.pred(x) for x in A])

        def fit(self, explanatory, target, verbose=0):
                """
                Initialise certains attributs de l'arbre puis appelle une nouvelle méthode
                Decision_Tree.fit_node sur la racine.
                """
                if self.split_criterion == "random":
                        self.split_criterion = self.random_split_criterion
                else:
                        self.split_criterion = self.Gini_split_criterion
                self.explanatory = explanatory
                self.target = target
                self.root.sub_population = np.ones_like(self.target, dtype='bool')

                self.fit_node(self.root)

                self.update_predict()
        
        if verbose == 1:
            # print("----------------------------------------------------")
            print(f"  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}")
            print(f"    - Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}")
            # print("----------------------------------------------------")

        def np_extrema(self, arr):
                """
                Calcule les valeurs minimale et maximale d'un tableau utilisant NumPy.
                Retourne les valeurs sous forme de tuple.
                """
                return np.min(arr), np.max(arr)

        def random_split_criterion(self, node):
                """
                Sélectionne aléatoirement une caractéristique et un seuil pour diviser
                la sous-population du nœud.

                Args:
                        node (Node): Le nœud à diviser.

                Retourne:
                        tuple: Un tuple contenant la caractéristique sélectionnée et le seuil.
                """
                diff = 0
                while diff == 0:
                        feature = self.rng.integers(0, self.explanatory.shape[1])
                        feature_min, feature_max = self.np_extrema(
                                self.explanatory[:, feature][node.sub_population])
                        diff = feature_max - feature_min
                x = self.rng.uniform()
                threshold = (1 - x) * feature_min + x * feature_max
                return feature, threshold

        def fit_node(self, node):
                """
                Ajuste un nœud de l'arbre de décision en divisant
                récursivement les données selon le meilleur critère de division.
                """
                node.feature, node.threshold = self.split_criterion(node)

                max_criterion = np.greater(
                        self.explanatory[:, node.feature],
                        node.threshold)

                left_population = np.logical_and(
                        node.sub_population,
                        max_criterion)

                # "La guerre ne détermine pas qui a raison - seulement qui reste."
                right_population = np.logical_and(
                        node.sub_population,
                        np.logical_not(max_criterion))

                # Le nœud gauche est-il une feuille ?
                is_left_leaf = np.any(np.array(
                        [node.depth == self.max_depth - 1,
                         np.sum(left_population) <= self.min_pop,
                         np.unique(self.target[left_population]).size == 1]))

                if is_left_leaf:
                        node.left_child = self.get_leaf_child(node, left_population)
                else:
                        node.left_child = self.get_node_child(node, left_population)
                        self.fit_node(node.left_child)

                # Le nœud droit est-il une feuille ?
                is_right_leaf = np.any(np.array(
                        [node.depth == self.max_depth - 1,
                         np.sum(right_population) <= self.min_pop,
                         np.unique(self.target[right_population]).size == 1]))

                if is_right_leaf:
                        node.right_child = self.get_leaf_child(node, right_population)
                else:
                        node.right_child = self.get_node_child(node, right_population)
                        self.fit_node(node.right_child)

        def get_leaf_child(self, node, sub_population):
                """
                Crée un nœud enfant feuille avec la valeur cible la plus
                fréquente dans la sous-population donnée et retourne l'objet nouveau.
                """
                value = np.argmax(np.bincount(self.target[sub_population]))
                leaf_child = Leaf(value)
                leaf_child.depth = node.depth + 1
                # NOTE cela devrait être leaf_child.subpopulation_leaf
                leaf_child.subpopulation = sub_population
                return leaf_child

        def get_node_child(self, node, sub_population):
                """
                Crée un nouveau nœud enfant pour le nœud parent donné.

                Args:
                        node (Node): Le nœud parent.
                        sub_population (list): La sous-population associée au nœud enfant.

                Retourne:
                        Node: Le nouveau nœud enfant créé.
                """
                n = Node()
                n.depth = node.depth + 1
                n.sub_population = sub_population
                return n

        def accuracy(self, test_explanatory, test_target):
                """
                Calcule la précision du modèle d'arbre de décision sur les données
                de test fournies.

                Args:
                test_explanatory (numpy.ndarray): Les variables explicatives des données
                        de test.
                test_target (numpy.ndarray): La variable cible des données de test.

                Retourne:
                float: La précision du modèle d'arbre de décision sur les données de test.
                """
                return np.sum(np.equal(
                        self.predict(test_explanatory), test_target)) / test_target.size

        def possible_thresholds(self, node, feature):
                """
                Calcule les seuils possibles pour un nœud et une caractéristique donnés.
                Retourne un numpy.ndarray de seuils possibles.
                """
                values = np.unique((self.explanatory[:, feature])[node.sub_population])
                return (values[1:] + values[:-1]) / 2

        def Gini_split_criterion_one_feature(self, node, feature):
                """
                Calcule le critère de division de Gini pour une caractéristique donnée
                dans un nœud.

                Args:
                        node (Node): Le nœud pour lequel calculer le critère de division de Gini.
                        feature (int): L'indice de la caractéristique à considérer.

                Retourne:
                        numpy.ndarray: Un tableau contenant le seuil avec l'impureté totale
                                la plus faible et la somme de Gini correspondante.
                """
                thresholds = self.possible_thresholds(node, feature)

                # Obtention du tableau des indices des individus dans la sous-population du nœud
                indices = np.arange(0, self.explanatory.shape[0])[node.sub_population]
                # Nombre d'indices, utilisé pour diviser l'expression plus tard
                div = indices.size

                # Obtention des valeurs de la caractéristique pour les individus dans la sous-population du nœud
                feature_values = (self.explanatory[:, feature])[node.sub_population]

                # Filtrer les tableaux pour les nœuds enfants gauche/droit
                filter_left = np.greater(feature_values[:, None], thresholds[None, :])
                filter_right = np.logical_not(filter_left)

                # Ne prendre que les individus dans la sous-population du nœud
                target_reduced = self.target[indices]

                # Classes uniques dans la sous-population du nœud
                classes = np.unique(self.target)

                # Calcul des masques de classe pour les enfants gauche/droit
                classes_mask = np.equal(target_reduced[:, None], classes)

                left_class_mask = np.logical_and(classes_mask[:, :, None],
                                                                                 filter_left[:, None, :])

                right_class_mask = np.logical_and(classes_mask[:, :, None],
                                                                                  filter_right[:, None, :])

                # Impuretés de Gini pour les enfants gauche et droit
                gini_left = 1 - np.sum(np.square(np.sum(left_class_mask, axis=0)),
                                                           axis=0) / (np.sum(filter_left, axis=0)) / div

                gini_right = 1 - np.sum(np.square(np.sum(right_class_mask, axis=0)),
                                                                axis=0) / (np.sum(filter_right, axis=0)) / div

                # Somme moyenne des impuretés de Gini
                gini_sum = gini_left + gini_right

                # Trouver l'indice du seuil avec l'impureté totale la plus faible
                gini_min = np.argmin(gini_left + gini_right)

                return np.array([thresholds[gini_min], gini_sum[gini_min]])

        def Gini_split_criterion(self, node):
                """
                Calcule le critère de division de Gini pour un nœud donné.

                Retourne:
                - i: L'indice de la caractéristique qui fournit la meilleure division.
                - gini: L'indice de Gini pour la meilleure division.
                """
                X = np.array(
                        [self.Gini_split_criterion_one_feature(node, i)
                         for i in range(self.explanatory.shape[1])])
                i = np.argmin(X[:, 1])

                return i, X[i, 0]
