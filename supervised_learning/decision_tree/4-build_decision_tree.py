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

    def max_depth_below(self):
        """
        Retourne la profondeur maximale de l'arbre sous ce nœud.
        """
        max_depth = self.depth

        # Si le nœud a un enfant gauche, calcule la profondeur maximale sous
        # l'enfant gauche
        if self.left_child is not None:
            max_depth = max(max_depth, self.left_child.max_depth_below())

        # Si le nœud a un enfant droit, calcule la profondeur maximale sous
        # l'enfant droit
        if self.right_child is not None:
            max_depth = max(max_depth, self.right_child.max_depth_below())

        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Compte les nœuds dans le sous-arbre enraciné à ce nœud.
        Optionnellement, compte uniquement les feuilles.
        """
        if only_leaves:
            # Si seules les feuilles doivent être comptées, saute le comptage
            # pour les nœuds non-feuilles.
            if self.is_leaf:
                return 1
            count = 0
        else:
            # Compte ce nœud si nous ne comptons pas uniquement les feuilles
            count = 1

        # Compte récursivement les nœuds dans les sous-arbres gauche et droit
        if self.left_child is not None:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child is not None:
            count += self.right_child.count_nodes_below(only_leaves)

        return count

    def __str__(self):
        """
        Retourne une représentation sous forme de chaîne de
        caractères du nœud et de ses enfants.
        """
        node_type = "racine" if self.is_root else "nœud"
        details = (f"{node_type} [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")
        if self.left_child:
            left_str = self.left_child.__str__().replace("\n", "\n    |  ")
            details += f"    +---> {left_str}"

        if self.right_child:
            right_str = self.right_child.__str__().replace("\n", "\n       ")
            details += f"\n    +---> {right_str}"

        return details

    def get_leaves_below(self):
        """
        Retourne une liste de toutes les feuilles sous ce nœud.
        """
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Calcule récursivement, pour chaque nœud, deux dictionnaires stockés
        comme attributs Node.lower et Node.upper. Ces dictionnaires
        contiennent les limites pour chaque caractéristique.
        """
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        if self.left_child:
            # Copie les limites du parent et met à jour
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()

            if self.feature in self.left_child.lower:
                # Met à jour la limite inférieure de l'enfant gauche pour la
                # caractéristique
                self.left_child.lower[self.feature] = max(
                    self.threshold, self.left_child.lower[self.feature]
                )
            else:
                self.left_child.lower[self.feature] = self.threshold

            # Recurse dans l'enfant gauche
            self.left_child.update_bounds_below()

        if self.right_child:
            # Copie les limites du parent et met à jour
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()

            if self.feature in self.right_child.upper:
                # Met à jour la limite supérieure de l'enfant droit pour la
                # caractéristique
                self.right_child.upper[self.feature] = min(
                    self.threshold, self.right_child.upper[self.feature]
                )
            else:
                self.right_child.upper[self.feature] = self.threshold

            # Recurse dans l'enfant droit
            self.right_child.update_bounds_below()


class Leaf(Node):
    """
    Classe représentant une feuille dans un arbre de décision.
    """

    def __init__(self, value, depth=None):
        """
        Initialise une feuille avec une valeur et une profondeur.

        Args:
            value: La valeur de prédiction de la feuille.
            depth (int, optional): La profondeur de la feuille dans l'arbre.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Retourne la profondeur de la feuille, car les feuilles
        sont les points finaux d'un arbre.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Retourne 1 car les feuilles comptent pour un nœud chacune.
        """
        return 1

    def __str__(self):
        """
        Retourne une représentation sous forme de
        chaîne de caractères de la feuille.
        """
        return f"-> feuille [value={self.value}] "

    def get_leaves_below(self):
        """
        Retourne une liste contenant uniquement cette feuille.
        """
        return [self]

    def update_bounds_below(self):
        """
        Les feuilles n'ont pas besoin de mettre à jour les limites car elles
        représentent les points finaux.
        """
        pass


class Decision_Tree():
    """
    Implémente un arbre de décision qui peut être utilisé pour divers
    processus de prise de décision.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialise l'arbre de décision avec des paramètres
        pour la construction de l'arbre et la génération
        de nombres aléatoires.

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
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Retourne la profondeur maximale d'un arbre.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Compte le nombre total de nœuds ou uniquement
        les nœuds feuilles dans l'arbre.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Retourne une représentation sous forme de chaîne de caractères de
        l'ensemble de l'arbre de décision.
        """
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """
        Récupère tous les nœuds feuilles de l'arbre.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Initie le processus de mise à jour des
        limites à partir de la racine.
        """
        self.root.update_bounds_below()
