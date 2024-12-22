#!/usr/bin/env python3
"""
Représente un nœud dans un arbre de décision.
"""
import numpy as np


class Node:
    """
    Classe représentant un nœud interne dans un arbre de décision.

    Attributs :
        feature (int or None): Indice de la caractéristique utilisée pour
            diviser les données.
        threshold (float or None): Valeur seuil pour la division.
        left_child (Node or None): Nœud enfant gauche.
        right_child (Node or None): Nœud enfant droit.
        is_leaf (bool): Indique si le nœud est une feuille.
        is_root (bool): Indique si le nœud est la racine de l'arbre.
        sub_population (np.ndarray or None): Masque booléen indiquant
            les échantillons dans ce nœud.
        depth (int): Profondeur du nœud dans l'arbre.
        lower (dict or None): Bornes inférieures pour les caractéristiques.
        upper (dict or None): Bornes supérieures pour les caractéristiques.
    """

    def __init__(
            self,
            feature=None,
            threshold=None,
            left_child=None,
            right_child=None,
            is_root=False,
            depth=0):
        """
        Initialise un nœud dans l'arbre de décision.

        Paramètres :
            feature (int or None, optionnel): Indice de la
                caractéristique utilisée pour diviser (default : None).
            threshold (float or None, optionnel): Valeur seuil pour
                la division (default : None).
            left_child (Node or None, optionnel): Nœud enfant gauche
                (default : None).
            right_child (Node or None, optionnel): Nœud enfant droit
                (default : None).
            is_root (bool, optionnel): Indique si le nœud est la racine
                (default : False).
            depth (int, optionnel): Profondeur du nœud dans l'arbre
                (default : 0).

        Retourne :
            None
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
        Calcule la profondeur maximale du sous-arbre enraciné à ce nœud.

        Retourne :
            int: La profondeur maximale du sous-arbre.
        """
        if self.is_leaf:
            return self.depth

        left_depth = self.depth
        if self.left_child:
            left_depth = self.left_child.max_depth_below()

        right_depth = self.depth
        if self.right_child:
            right_depth = self.right_child.max_depth_below()

        return max(left_depth, right_depth)


class Leaf(Node):
    """
    Classe représentant un nœud feuille dans un arbre de décision.

    Attributs :
        value (int or any): Valeur prédite par la feuille.
        depth (int or None): Profondeur de la feuille dans l'arbre.
    """

    def __init__(self, value, depth=None):
        """
        Initialise une feuille avec une valeur prédite.

        Paramètres :
            value (int or any): Valeur prédite par la feuille.
            depth (int or None, optionnel): Profondeur de la feuille dans
                l'arbre (default : None).

        Retourne :
            None
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Retourne la profondeur de la feuille.

        Retourne :
            int: La profondeur de la feuille.
        """
        return self.depth


class Decision_Tree:
    """
    Classe représentant un arbre de décision pour la classification.

    Attributs :
        rng (np.random.Generator): Générateur de nombres aléatoires
            pour la reproductibilité.
        root (Node): Nœud racine de l'arbre.
        explanatory (np.ndarray or None): Données explicatives
            d'entraînement.
        target (np.ndarray or None): Cibles d'entraînement.
        max_depth (int): Profondeur maximale de l'arbre.
        min_pop (int): Population minimale requise pour diviser
            un nœud.
        split_criterion (str): Critère de division des nœuds
            ("random" ou autre).
        predict (callable or None): Fonction de prédiction basée sur l'arbre.
    """

    def __init__(
            self,
            max_depth=10,
            min_pop=1,
            seed=0,
            split_criterion="random",
            root=None):
        """
        Initialise un arbre de décision avec les paramètres spécifiés.

        Paramètres :
            max_depth (int, optionnel): Profondeur maximale de l'arbre
                (default : 10).
            min_pop (int, optionnel): Population minimale pour diviser
                un nœud (default : 1).
            seed (int, optionnel): Graine pour le générateur aléatoire
                (default : 0).
            split_criterion (str, optionnel): Critère de division des
                nœuds (default : "random").
            root (Node or None, optionnel): Nœud racine de l'arbre
                (default : None).

        Retourne :
            None
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
        Calcule la profondeur maximale de l'arbre.

        Retourne :
            int: Profondeur maximale de l'arbre.
        """
        return self.root.max_depth_below()
