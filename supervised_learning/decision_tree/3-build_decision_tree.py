#!/usr/bin/env python3
"""
Module implémentant les classes pour construire et manipuler un arbre de décision.
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

    def get_leaves_below(self):
        """
        Retourne la liste de toutes les feuilles sous ce nœud.

        Returns:
            list: Liste des feuilles sous ce nœud.
        """
        if self.is_leaf:
            return [self]
        
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        
        return leaves


class Leaf(Node):
    """
    Classe représentant une feuille dans un arbre de décision.
    """
    def __init__(self, value, depth=None):
        """
        Initialise une feuille de l'arbre de décision.

        Args:
            value: La valeur de prédiction de la feuille.
            depth (int, optional): La profondeur de la feuille dans l'arbre.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def get_leaves_below(self):
        """
        Retourne une liste contenant uniquement cette feuille.

        Returns:
            list: Liste contenant cette feuille.
        """
        return [self]

    def __str__(self):
        """
        Retourne une représentation textuelle de la feuille.

        Returns:
            str: Représentation textuelle de la feuille.
        """
        return f"-> leaf [value={self.value}]"


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
            root (Node): Nœud racine de l'arbre.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        
    def get_leaves(self):
        """
        Retourne la liste de toutes les feuilles de l'arbre.

        Returns:
            list: Liste de toutes les feuilles de l'arbre.
        """
        return self.root.get_leaves_below()
