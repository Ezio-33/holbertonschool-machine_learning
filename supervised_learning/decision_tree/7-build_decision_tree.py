#!/usr/bin/env python3
"""
Module
"""
import numpy as np


class Node:
    """
    Args: ff
    Returns: ff
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

    def __str__(self):
        """
        Args: ff
        Returns: ff
        """
        p = "root" if self.is_root else "-> node"
        result = f"{p} [feature={self.feature},\
 threshold={self.threshold}]\n"
        if self.left_child:
            result +=\
                self.left_child_add_prefix(self.left_child.__str__().strip())
        if self.right_child:
            result +=\
                self.right_child_add_prefix(self.right_child.__str__().strip())
        return result

    def left_child_add_prefix(self, text):
        """
        Args: ff

        Returns: ff
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """
        Args: ff
        Returns: ff
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def max_depth_below(self):
        """
        Args: ff
        Returns: ff
        """
        max_depth = self.depth
        if self.left_child:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child:
            max_depth = max(max_depth, self.right_child.max_depth_below())
        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Args: ff
        Returns: ff
        """
        if only_leaves:
            if self.is_leaf:
                return 1
            count = 0
        else:
            count = 1

        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def get_leaves_below(self):
        """
        Args: ff
        Returns: ff
        """
        leaves = []
        if self.is_leaf:
            leaves.append(self)
        else:
            if self.left_child:
                leaves.extend(self.left_child.get_leaves_below())
            if self.right_child:
                leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Args: ff
        Returns: ff
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                child.upper = self.upper.copy()
                child.lower = self.lower.copy()

                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                elif child == self.right_child:
                    child.upper[self.feature] = self.threshold
        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """
        Args: ff
        Returns: ff
        """
        def is_large_enough(x):
            return np.array([np.greater_equal(x[:, key], self.lower[key])
                            for key in self.lower.keys()]).all(axis=0)

        def is_small_enough(x):
            return np.array([np.less_equal(x[:, key], self.upper[key])
                            for key in self.upper.keys()]).all(axis=0)

        self.indicator = lambda x: np.logical_and(is_large_enough(x),
                                                  is_small_enough(x))

    def update_predict(self):
        """
        Args: ff
        Returns: ff
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([leaf.value
                                           for x in A for leaf in leaves
                                           if leaf.indicator(x)])

    def pred(self, x):
        """
        Prédit la valeur pour un échantillon en déléguant au nœud racine
        de l'arbre.

        Args:
            x (array): Les caractéristiques d'entrée pour un échantillon.

        Return:
            any: La valeur prédite de l'arbre.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    Args: ff
    Returns: ff
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """
        Args: ff
        Returns: ff
        """
        return (f"-> leaf [value={self.value}] ")

    def max_depth_below(self):
        """
        Args: ff
        Returns: ff
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Args: ff
        Returns: ff
        """
        return 1

    def get_leaves_below(self):
        """
        Args: ff
        Returns: ff
        """
        return [self]

    def update_bounds_below(self):
        """
        Args: ff
        Returns: ff
        """
        pass

    def pred(self, x):
        """
        Args: ff
        Returns: ff
        """
        return self.value


class Decision_Tree():
    """
    Args: ff
    Returns: ff
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def __str__(self):
        """
        Args: ff
        Returns: ff
        """
        return self.root.__str__()

    def depth(self):
        """
        Args: ff
        Returns: ff
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Args: ff
        Returns: ff
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """
        Args: ff
        Returns: ff
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Args: ff
        Returns: ff
        """
        self.root.update_bounds_below()

    def update_bounds(self):
        """
        Args: ff
        Returns: ff
        """
        self.root.update_bounds_below()

    def update_indicator(self):
        """
        Args: ff
        Returns: ff
        """
        self.root.update_indicator()

    def pred(self, x):
        """
        Args: ff
        Returns: ff
        """
        return self.root.pred(x)

    def pred(self, x):
        """
        Prédit la valeur pour un échantillon en déléguant au nœud racine
        de l'arbre.

        Args:
            x (array): Les caractéristiques d'entrée pour un échantillon.

        Return:
            any: La valeur prédite de l'arbre.
        """
        return self.root.pred(x)

    def update_predict(self):
        """
        Met à jour la fonction de prédiction pour l'arbre de décision.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([
            next(leaf.value for leaf in leaves
                 if leaf.indicator(x.reshape(1, -1)))
            for x in np.atleast_2d(A)
        ])

    def fit(self, explanatory, target, verbose=0):
        """
        Ajuste l'arbre de décision aux données d'entraînement fournies.

        Args:
            explanatory (array): Caractéristiques d'entrée pour les données
            d'entraînement.
            target (array): Valeurs cibles pour les données d'entraînement.
            verbose (int, optionnel): Niveau de verbosité de la sortie.
            Par défaut est 0.

        Cette méthode configure l'arbre basé sur le critère de division choisi
        et ajuste récursivement les nœuds, en mettant à jour la fonction de
        prédiction de l'arbre une fois terminé.
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
            print(f"  Entraînement terminé.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}")
            print(f"    - Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}")
            # print("----------------------------------------------------")

    def np_extrema(self, arr):
        """
        Calcule les valeurs minimales et maximales d'un tableau.

        Args:
            arr (array): Le tableau d'entrée.

        Return:
            tuple: Les valeurs minimale et maximale dans le tableau.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Détermine un critère de division aléatoire pour un nœud basé
        sur les valeurs des caractéristiques.

        Args:
            node (Node): Le nœud pour lequel déterminer la division.

        Return:
            tuple: L'indice de la caractéristique choisie et la valeur seuil
            pour la division.
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
        Ajuste récursivement l'arbre à partir du nœud donné.

        Args:
            node (Node): Le nœud à partir duquel commencer l'ajustement
            de l'arbre.

        Cette méthode divise le nœud si les conditions le permettent, ou le
        convertit en feuille si les conditions de division ne sont pas
        remplies (basé sur la profondeur, la population ou la pureté).
        """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & ~left_population

        # La gauche est-elle une feuille ?
        is_left_leaf = (node.depth == self.max_depth - 1 or
                        np.sum(left_population) <= self.min_pop or
                        np.unique(self.target[left_population]).size == 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            node.left_child.depth = node.depth + 1
            self.fit_node(node.left_child)

        # La droite est-elle une feuille ?
        is_right_leaf = (node.depth == self.max_depth - 1 or
                         np.sum(right_population) <= self.min_pop or
                         np.unique(self.target[right_population]).size == 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            node.right_child.depth = node.depth + 1
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Crée un nœud feuille à partir de la sous-population donnée.

        Args:
            node (Node): Le nœud parent dont la feuille est dérivée.
            sub_population (array): Sous-ensemble d'indices indiquant la
            population pour la feuille.

        Return:
            Leaf: Un nouveau nœud feuille avec une valeur déterminée par la
            classe la plus commune dans sub_population.
        """
        target_values = self.target[sub_population]
        values, counts = np.unique(target_values, return_counts=True)
        value = values[np.argmax(counts)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Crée un nouveau nœud enfant pour des divisions supplémentaires.

        Args:
            node (Node): Le nœud parent.
            sub_population (array): Sous-ensemble d'indices pour la
            population du nouveau nœud.

        Return:
            Node: Un nouveau nœud enfant initialisé pour des divisions
            supplémentaires.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        Calcule la précision du modèle de prédiction sur les données de test.

        Args:
            test_explanatory (array): Les variables explicatives des données
            de test.
            test_target (array): Les variables cibles des données de test.

        Return:
            float: La précision du modèle sur les données de test,
            calculée comme le ratio des prédictions correctes.
        """
        preds = self.predict(test_explanatory) == test_target
        return np.sum(preds) / len(test_target)
