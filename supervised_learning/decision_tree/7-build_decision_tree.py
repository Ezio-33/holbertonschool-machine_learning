#!/usr/bin/env python3

"""
Composants de l'Arbre de Décision
"""
import numpy as np


class Node:
    """
    Représente un nœud de décision dans un arbre de décision,
    qui peut diviser les donnée en fonction des fonctionnalités et des seuils.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialise le nœud avec des séparations de fonctionnalités
        optionnelles, des valeurs de seuil, des enfants,
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
        self.lower = None  # Initialise à None, à mettre à jour
        self.upper = None  # Initialise à None, à mettre à jour

    def max_depth_below(self):
        """
        Retourne la profondeur maximale de l'arbre en dessous de ce nœud.
        """
        max_depth = self.depth

        # Si le nœud a un enfant gauche, calcule la profondeur
        # maximale en dessous de l'enfant gauche
        if self.left_child is not None:
            max_depth = max(max_depth, self.left_child.max_depth_below())

        # Si le nœud a un enfant droit, calcule la profondeur
        # maximale en dessous de l'enfant droit
        if self.right_child is not None:
            max_depth = max(max_depth, self.right_child.max_depth_below())

        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Compte les nœuds dans le sous-arbre enraciné à ce nœud.
        Optionnellement, compte uniquement les nœuds feuilles.
        """
        if only_leaves:
            # Si seuls les feuilles doivent être comptés, ne compte pas les
            # nœuds non-feuilles.
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
        Retourne une représentation en chaîne du nœud et de ses enfants
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
        Retourne une liste de toutes les feuilles en dessous de ce nœud.
        """
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Calcule récursivement, pour chaque nœud, deux dictionnaires
        stockés comme attributs Node.lower et Node.upper.
        Ces dictionnaires contiennent les limites
        pour chaque fonctionnalité.
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
                # fonctionnalité
                self.left_child.lower[self.feature] = max(
                    self.threshold, self.left_child.lower[self.feature]
                )
            else:
                self.left_child.lower[self.feature] = self.threshold

            # Récursion dans l'enfant gauche
            self.left_child.update_bounds_below()

        if self.right_child:
            # Copie les limites du parent et met à jour
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()

            if self.feature in self.right_child.upper:
                # Met à jour la limite supérieure de l'enfant droit pour la
                # fonctionnalité
                self.right_child.upper[self.feature] = min(
                    self.threshold, self.right_child.upper[self.feature]
                )
            else:
                self.right_child.upper[self.feature] = self.threshold

            # Récursion dans l'enfant droit
            self.right_child.update_bounds_below()

    def update_indicator(self):
        """
        Met à jour la fonction indicatrice basée sur les limites
        inférieures et supérieures.
        """
        def is_large_enough(x):
            """
            est suffisamment grand
            """
            comparisons = [x[:, key] > self.lower[key] for key in self.lower]
            return np.all(comparisons, axis=0)

        def is_small_enough(x):
            """
            est suffisamment petit
            """
            comparisons = [x[:, key] <= self.upper[key] for key in self.upper]
            return np.all(comparisons, axis=0)

        self.indicator = lambda x: (
            np.logical_and(is_large_enough(x), is_small_enough(x))
        )

    def pred(self, x):
        """
        Prédit l'étiquette de classe pour une instance unique x
        basée sur la structure de l'arbre
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    Représente un nœud feuille dans un arbre de décision,
    contenant une valeur constante et une profondeur.
    """

    def __init__(self, value, depth=None):
        """
        Initialise la feuille avec une valeur
        spécifique et une profondeur.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Retourne la profondeur de la feuille, car les nœuds
        feuilles sont les points terminaux d'un arbre
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Retourne 1 car chaque feuille compte comme un nœud.
        """
        return 1

    def __str__(self):
        """
        Retourne une représentation en chaîne de la feuille.
        """
        return f"-> feuille [value={self.value}] "

    def get_leaves_below(self):
        """
        Retourne une liste contenant uniquement cette feuille.
        """
        return [self]

    def update_bounds_below(self):
        """
        Les feuilles n'ont pas besoin de mettre à jour les limites
        car elles représentent des points terminaux
        """
        pass

    def pred(self, x):
        """
        Prédit la valeur de la feuille
        """
        return self.value


class Decision_Tree():
    """
    Implémente un arbre de décision qui peut être utilisé pour divers
    processus de prise de décision.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialise l'arbre de décision avec des paramètres
        pour la construction de l'arbre et
        la génération de nombres aléatoires.
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
        Retourne la profondeur maximale d'un arbre
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Compte le nombre total de nœuds ou seulement
        les nœuds feuilles dans l'arbre
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Retourne une représentation en chaîne de l'ensemble
        de l'arbre de décision.
        """
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """
        Récupère tous les nœuds feuilles de l'arbre.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Initialise le processus de mise à jour des
        limites à partir de la racine.
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Met à jour la fonction de prédiction pour l'arbre de décision.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict(A):
            results = np.empty(A.shape[0], dtype=int)
            for leaf in leaves:
                indices = leaf.indicator(A)
                results[indices] = leaf.value
            return results

        self.predict = predict

    def pred(self, x):
        """
        Prédit l'étiquette de classe pour une instance unique x
        """
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        """
        Initialise l'entraînement en configurant le nœud racine
        et les critères de séparation puis effectue l'ajustement
        récursif des nœuds et met à jour le modèle de prédiction.
        """
        # Choisir le critère de séparation basé sur la configuration
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.gini_split_criterion

            def gini_split_criterion(self, node):
                # Implementation of Gini split criterion
                pass

        # Assigner les données aux attributs de l'arbre
        self.explanatory = explanatory
        self.target = target
        # Commencer avec tous les échantillons à la racine
        self.root.sub_population = np.ones_like(self.target, dtype=bool)

        # Commencer la construction récursive de l'arbre
        self.fit_node(self.root)

        # Préparer l'arbre pour faire des prédictions
        self.update_predict()

        # Afficher le résumé de l'entraînement si verbose
        if verbose == 1:
            print(f"""  Entraînement terminé.
- Profondeur                : { self.depth()       }
- Nombre de nœuds          : { self.count_nodes() }
- Nombre de feuilles        : { self.count_nodes(only_leaves=True) }
- Précision sur les données d'entraînement : { self.accuracy(self.explanatory,
                                                  self.target)}""")

    def np_extrema(self, arr):
        """
        Retourne le minimum et le maximum d'un tableau
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Détermine aléatoirement une fonctionnalité et
        un seuil pour diviser le nœud.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            sub_pop = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(sub_pop)
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """
        Ajuste récursivement un nœud, divisant les données et créant
        des nœuds enfants si nécessaire
        """
        node.feature, node.threshold = self.split_criterion(node)

        # Masques booléens pour les sous-populations gauche et droite
        feature = self.explanatory[:, node.feature]
        left_population = node.sub_population & (feature > node.threshold)
        right_population = node.sub_population & ~left_population

        # Vérifier les conditions pour que l'enfant gauche soit une feuille
        is_left_leaf = (node.depth == self.max_depth - 1 or
                        np.sum(left_population) <= self.min_pop or
                        np.unique(self.target[left_population]).size == 1)

        # Créer le nœud enfant gauche ou une feuille
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Vérifier les conditions pour que l'enfant droit soit une feuille
        is_right_leaf = (node.depth == self.max_depth - 1 or
                         np.sum(right_population) <= self.min_pop or
                         np.unique(self.target[right_population]).size == 1)

        # Créer le nœud enfant droit ou une feuille
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Crée un nœud feuille en utilisant la classe la plus fréquente dans
        la sous-population
        """
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Crée un nœud enfant non-feuille.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        Calcule la précision de l'arbre de décision sur les données de test
        """
        predictions = self.predict(test_explanatory)
        return np.mean(predictions == test_target)
