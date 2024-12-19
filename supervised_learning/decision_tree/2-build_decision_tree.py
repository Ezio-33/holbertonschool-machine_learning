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

    def left_child_add_prefix(self, text):
        """
        Ajoute les préfixes pour l'enfant gauche.

        Args:
            Représentation textuelle de l'enfant gauche.

        Return:
            Texte avec préfixes ajoutés.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Ajoute les préfixes pour l'enfant droit.

        Args:
            Représentation textuelle de l'enfant droit.

        Return:
            Texte avec préfixes ajoutés.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def __str__(self):
        """
        Return:
            Représentation textuelle du nœud.
        """
        node_type = "root" if self.is_root else "node"
        details = f"{node_type} [feature={self.feature},"
        details += f" threshold={self.threshold}]\n"

        if self.left_child:
            left_str = self.left_child.__str__().replace("\n", "\n    |  ")
            details += f"    +---> {left_str}"

        if self.right_child:
            right_str = self.right_child.__str__().replace("\n", "\n       ")
            details += f"\n    +---> {right_str}"

        return details.rstrip()


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

    def __str__(self):
        """
        Return:
            Représentation textuelle de la feuille.
        """
        return f"leaf [value={self.value}]"


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

    def __str__(self):
        """
        Return:
            Représentation textuelle complète de l'arbre.
        """
        return self.root.__str__() + "\n"
