#!/usr/bin/env python3
"""
Module implémentant les classes pour construire et
manipuler un arbre de décision.
"""

import numpy as np


class Node:
    """
    Classe représentant un nœud dans un arbre de décision.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialise un nœud de l'arbre de décision.

        Args:
            feature (int, optional): L'indice de la caractéristique utilisée.
            threshold (float, optional): La valeur seuil pour la division.
            left_child (Node, optional): L'enfant gauche du nœud.
            right_child (Node, optional): L'enfant droit du nœud.
            is_root (bool, optional): Indique si le nœud est la racine.
            depth (int, optional): La profondeur du nœud dans l'arbre.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def count_nodes_below(self, only_leaves=False):
        """
        Compte le nombre de nœuds ou de feuilles sous ce nœud.

        Args:
            only_leaves (bool): Si True, ne compte que les feuilles.

        Return:
            int: Le nombre de nœuds ou de feuilles.
        """
        if self.is_leaf:
            return 1

        count = 0 if only_leaves else 1
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)

        return count


class Leaf(Node):
    """
    Classe représentant une feuille dans un arbre de décision.
    """

    def __init__(self, value, depth=None):
        """
        Initialise une feuille de l'arbre de décision.

        Args:
            value: La valeur de prédiction de la feuille.
            depth (int, optional): La profondeur de
            la feuille dans l'arbre.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def count_nodes_below(self, only_leaves=False):
        """
        Compte le nombre de nœuds sous cette feuille (toujours 1).

        Args:
            only_leaves (bool): Si True, ne compte que les feuilles.

        Return:
            Toujours 1 car une feuille est un nœud terminal.
        """
        return 1


class Decision_Tree:
    """
    Classe représentant un arbre de décision complet.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialise un arbre de décision.

        Args:
            max_depth (int): Profondeur maximale de l'arbre.
            min_pop (int): Population minimale pour un nœud.
            seed (int): Graine pour la reproduction des résultats.
            split_criterion (str): Critère de division des nœuds.
            root (Node, optional): Nœud racine de l'arbre.
        """
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

    def count_nodes(self, only_leaves=False):
        """
        Compte le nombre total de nœuds ou de feuilles dans l'arbre.

        Args:
            only_leaves (bool): Si True, ne compte que les feuilles.

        Return:
            Le nombre de nœuds ou de feuilles.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
