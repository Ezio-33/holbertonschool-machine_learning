#!/usr/bin/env python3
"""
Module implémentant les classes pour construire et
manipuler un arbre de décision.
"""
import numpy as np


class Node:
    """
    Représente un nœud de décision dans un arbre de décision,
    qui peut diviser les données en fonction des
    caractéristiques et des seuils.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialise le nœud avec des divisions de caractéristiques
        optionnelles, des valeurs de seuil,des enfants,
        le statut de racine et la profondeur.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = None
        self.upper = None

    def max_depth_below(self):
        """
        Renvoie la profondeur maximale de l'arbre sous ce nœud.
        """
        max_depth = self.depth

        if self.left_child is not None:
            max_depth = max(max_depth, self.left_child.max_depth_below())

        if self.right_child is not None:
            max_depth = max(max_depth, self.right_child.max_depth_below())

        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Compte les nœuds dans le sous-arbre enraciné à ce nœud.
        Optionnellement, compte uniquement les nœuds feuilles.
        """
        if only_leaves:
            if self.is_leaf:
                return 1
            count = 0
        else:
            count = 1

        if self.left_child is not None:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child is not None:
            count += self.right_child.count_nodes_below(only_leaves)

        return count

    def __str__(self):
        """
        Returns a string representation of the node and its children.
        """
        node_type = "root" if self.is_root else "node"
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
        Renvoie une liste de toutes les feuilles sous ce nœud.
        """
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Calcule récursivement, pour chaque nœud,
        deux dictionnaires stockés en tant qu'attributs
        Node.lower et Node.upper. Ces dictionnaires
        contiennent les limites pour chaque caractéristique.
        """
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        if self.left_child:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()

            if self.feature in self.left_child.lower:
                self.left_child.lower[self.feature] = max(
                    self.threshold, self.left_child.lower[self.feature]
                )
            else:
                self.left_child.lower[self.feature] = self.threshold

            self.left_child.update_bounds_below()

        if self.right_child:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()

            if self.feature in self.right_child.upper:
                self.right_child.upper[self.feature] = min(
                    self.threshold, self.right_child.upper[self.feature]
                )
            else:
                self.right_child.upper[self.feature] = self.threshold

            self.right_child.update_bounds_below()

    def update_indicator(self):
        """
        Met à jour la fonction indicatrice basée
        sur les limites inférieures et supérieures.
        """
        def is_large_enough(x):
            comparisons = [x[:, key] > self.lower[key] for key in self.lower]
            return np.all(comparisons, axis=0)

        def is_small_enough(x):
            comparisons = [x[:, key] <= self.upper[key] for key in self.upper]
            return np.all(comparisons, axis=0)

        self.indicator = lambda x: (
            np.logical_and(is_large_enough(x), is_small_enough(x))
        )


class Leaf(Node):
    """
    Représente un nœud feuille dans un arbre de décision,
    contenant une valeur constante et une profondeur.
    """

    def __init__(self, value, depth=None):
        """
        Initialise la feuille avec une valeur et une profondeur spécifiques.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Renvoie la profondeur de la feuille, car les nœuds
        de la feuille sont les points d'extrémité d'un arbre.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Renvoie 1 car les feuilles comptent pour un nœud chacune.
        """
        return 1

    def __str__(self):
        """
        Renvoie une représentation sous forme
        de chaîne de caractères de la feuille.
        """
        return f"-> leaf [value={self.value}] "

    def get_leaves_below(self):
        """
        Renvoie une liste ne contenant que cette feuille.
        """
        return [self]

    def update_bounds_below(self):
        """
        Les feuilles n'ont pas besoin de mettre à jour
        les limites car elles représentent des points d'extrémité.
        """
        pass


class Decision_Tree:
    """
    Implémente un arbre de décision qui peut être
    utilisé pour divers processus de prise de décision.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialise l'arbre de décision avec des paramètres
        pour la construction de l'arbre et la génération
        de nombres aléatoires.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)

    def depth(self):
        """
        Retourne la profondeur maximale de l'arbre.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Compte le nombre total de nœuds ou seulement
        les nœuds feuilles dans l'arbre.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Retourne une représentation sous forme de chaîne de
        caractères de l'arbre de décision entier.
        """
        return str(self.root) + "\n"

    def get_leaves(self):
        """
        Récupère tous les nœuds feuilles de l'arbre.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Initie le processus de mise à jour des limites à partir de la racine.
        """
        self.root.update_bounds_below()
