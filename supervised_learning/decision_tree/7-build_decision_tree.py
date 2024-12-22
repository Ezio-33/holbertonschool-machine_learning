a#!/usr/bin/env python3
""" Tâche 7"""
import numpy as np


def left_child_add_prefix(text):
    """
    Ajoute un préfixe à chaque ligne du texte pour
    indiquer qu'il s'agit de l'enfant de gauche dans l'arborescence.

    Paramètres:
    text : str
        Texte auquel le préfixe sera ajouté.

    Retourne:
    str
        Le texte avec le préfixe enfant gauche ajouté à chaque ligne.
    """
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("    |  " + x) + "\n"
    return new_text


def right_child_add_prefix(text):
    """
    Ajoute un préfixe à chaque ligne du texte pour indiquer
    qu'il s'agit de l'enfant de droite dans l'arborescence.

    Paramètres:
    text : str
        Texte auquel le préfixe sera ajouté.
    Retourne:
    str
        Le texte avec le préfixe enfant de droite ajouté à chaque ligne.
    """
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("       " + x) + "\n"
    return new_text


class Node:
    """
    Classe représentant un nœud dans un arbre de décision.

    Attributs :
    feature : int ou None
        La caractéristique utilisée pour diviser les données.
    threshold : float ou None
        La valeur seuil pour la division.
    left_child : Node ou None
        Le nœud enfant gauche.
    right_child : Node ou None
        Le nœud enfant de droite.
    is_leaf : bool
        Booléen indiquant si le nœud est une feuille.
    is_root : bool
        Booléen indiquant si le nœud est la racine.
    sub_population : any
        Le sous-ensemble de données de ce nœud.
    depth : int
        La profondeur du nœud dans l'arbre.

    Méthodes :
    max_depth_below() :
        Calcule la profondeur maximale du sous-arbre dont le
        point d'ancrage est ce nœud.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """
        Initialise un nœud avec les paramètres donnés.

        Paramètres :
        feature : int ou None, optionnel
            La caractéristique utilisée pour diviser les données
            (par défaut, None).
        threshold : float ou None, optionnel
            La valeur seuil pour le découpage
            (par défaut : None).
        left_child : Node ou None, optionnel
            Le nœud enfant gauche (par défaut, None).
        right_child : Node ou None, optionnel
            Le nœud enfant de droite (par défaut, None).
        is_root : bool, optionnel
            Booléen indiquant si le nœud est la racine
            (False par défaut).
        depth : int, optionnel
            La profondeur du nœud dans l'arbre (0 par défaut).
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
        Calcule la profondeur maximale du sous-arbre enraciné à ce nœud.

        Retourne :
        int
            La profondeur maximale du sous-arbre.
        """
        if self.is_leaf:
            return self.depth
        if self.left_child:
            left_depth = self.left_child.max_depth_below()
        else:
            left_depth = self.depth
        if self.right_child:
            right_depth = self.right_child.max_depth_below()
        else:
            right_depth = self.depth
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Compte le nombre de nœuds dans le sous-arbre enraciné à ce nœud.

        Paramètres :
        only_leaves : bool, optionnel
            Si True, ne compte que les nœuds feuilles (False par défaut).

        Retourne :
        int
            Le nombre de nœuds dans le sous-arbre.
        """
        if only_leaves:
            # Compte les feuilles dans les deux enfants
            return (
                self.left_child.count_nodes_below(only_leaves=True) +
                self.right_child.count_nodes_below(only_leaves=True)
            )
        else:
            # Compte tous les nœuds dans le sous-arbre
            return (
                1 + self.left_child.count_nodes_below(only_leaves=False) +
                self.right_child.count_nodes_below(only_leaves=False)
            )

    def __str__(self):
        """
        Fournit une représentation sous forme de chaîne de caractères du nœud,
        y compris ses enfants.

        Retourne :
        chaîne
            Une chaîne formatée représentant le sous-arbre enraciné dans ce nœud.
        """
        if self.is_root:
            result = (
                f"root [feature={self.feature}, threshold={self.threshold}]\n"
            )
        else:
            result = (
                f"node [feature={self.feature}, threshold={self.threshold}]\n"
            )

        # Ajoute l'enfant gauche avec préfixe
        if self.left_child:
            left_str = self.left_child.__str__()
            result += left_child_add_prefix(left_str)

        # Ajoute l'enfant droit avec préfixe
        if self.right_child:
            right_str = self.right_child.__str__()
            result += right_child_add_prefix(right_str)

        return result

    def get_leaves_below(self):
        """
        Retourne la liste de tous les nœuds feuilles dans le sous-arbre
        enraciné à ce nœud.

        Retourne :
        list
            Une liste de toutes les feuilles dans le sous-arbre.
        """
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Calcule et met à jour récursivement les dictionnaires de
        bornes inférieures et supérieures pour chaque nœud et ses
        enfants en fonction des seuils des caractéristiques.
        """
        if self.is_root:
            # Initialise les bornes à la racine
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        # Calcule les bornes pour les enfants
        if self.left_child:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()
            # Met à jour la borne supérieure pour la caractéristique
            self.left_child.lower[self.feature] = self.threshold

        if self.right_child:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()
            # Met à jour la borne inférieure pour la caractéristique
            self.right_child.upper[self.feature] = self.threshold

        # Met à jour récursivement les bornes pour les enfants
        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """
        Calcule la fonction indicatrice pour le nœud actuel
        basée sur les bornes inférieures et supérieures.
        """

        def is_large_enough(x):
            """
            Vérifie si chaque individu a toutes ses caractéristiques
            supérieures aux bornes inférieures.

            Paramètres:
            x : np.ndarray
                Un tableau NumPy 2D de forme (n_individus, n_caractéristiques).

            Retourne:
            np.ndarray
                Un tableau 1D de valeurs booléennes indiquant si chaque individu
                satisfait la condition.
            """
            return np.all(np.array([x[:, key] > self.lower[key]
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(x):
            """
            Vérifie si chaque individu a toutes ses caractéristiques
            inférieures ou égales aux bornes supérieures.

            Paramètres:
            x : np.ndarray
                Un tableau NumPy 2D de forme (n_individus, n_caractéristiques).

            Retourne:
            np.ndarray
                Un tableau 1D de valeurs booléennes indiquant
            """
            return np.all(np.array([x[:, key] <= self.upper[key]
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda x: \
            np.all(np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """
        Prédit la classe pour un individu unique au nœud.

        Paramètres:
        x : np.ndarray
            Un tableau NumPy 1D représentant les caractéristiques
            d'un individu unique.

        Retourne:
        int
            La classe prédite pour l'individu.
        """
        if self.is_leaf:
            return self.value
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    Classe représentant un nœud feuille dans un arbre de décision,
    héritant de Node.

    Attributs:
    value : any
        La valeur prédite par la feuille.
    depth : int
        La profondeur de la feuille dans l'arbre.

    Méthodes:
    max_depth_below():
        Retourne la profondeur de la feuille.
    """

    def __init__(self, value, depth=None):
        """
        Initialise une feuille avec les paramètres donnés.

        Paramètres:
        value : any
            La valeur prédite par la feuille.
        depth : int, optionnel
            La profondeur de la feuille dans l'arbre (par défaut, None).
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Retourne la profondeur de la feuille.

        Retourne:
        int
            La profondeur de la feuille.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Compte le nombre de nœuds dans le sous-arbre enraciné à cette feuille.

        Paramètres:
        only_leaves : bool, optionnel
            Si True, compte seulement les feuilles (False par défaut).

        Retourne:
        int
            Le nombre de nœuds dans le sous-arbre.
        """
        return 1

    def __str__(self):
        """
        Retourne une représentation en chaîne de caractères du nœud feuille.

        Retourne:
        str
            La représentation en chaîne de la feuille.
        """
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """
        Retourne la feuille elle-même dans une liste.

        Retourne:
        list
            Une liste contenant cette feuille.
        """
        return [self]

    def update_bounds_below(self):
        """
        Les nœuds feuille héritent des bornes de leurs
        nœuds parents et ne propagent pas plus loin.
        """
        # Les bornes sont héritées du nœud parent et restent inchangées
        pass

    def pred(self, x):
        """
        Prédit la classe pour un individu unique au nœud feuille.

        Paramètres:
        x : np.ndarray
            Un tableau NumPy 1D représentant les caractéristiques
            d'un individu unique.

        Retourne:
        int
            La classe prédite pour l'individu.
        """
        return self.value


class Decision_Tree():
    """
    Classe représentant un arbre de décision.

    Attributs:
    rng : numpy.random.Generator
        Générateur de nombres aléatoires pour la reproductibilité.
    root : Node
        Le nœud racine de l'arbre.
    explanatory : any
        Les caractéristiques explicatives du jeu de données.
    target : any
        Les valeurs cibles du jeu de données.
    max_depth : int
        La profondeur maximale de l'arbre.
    min_pop : int
        La population minimale requise pour diviser un nœud.
    split_criterion : str
        Le critère utilisé pour diviser les nœuds.
    predict : any
        Méthode pour prédire la valeur cible pour un ensemble de caractéristiques.

    Méthodes:
    depth():
        Retourne la profondeur maximale de l'arbre.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialise un arbre de décision avec les paramètres donnés.

        Paramètres:
        max_depth : int, optionnel
            La profondeur maximale de l'arbre (par défaut 10).
        min_pop : int, optionnel
            La population minimale requise pour diviser un nœud
            (par défaut 1).
        seed : int, optionnel
            Graine pour le générateur aléatoire (par défaut 0).
        split_criterion : str, optionnel
            Le critère utilisé pour diviser les nœuds
            (par défaut "random").
        root : Node ou None, optionnel
            Le nœud racine de l'arbre (par défaut None).
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
        Retourne la profondeur maximale de l'arbre.

        Retourne:
        int
            La profondeur maximale de l'arbre.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Compte le nombre de nœuds dans l'arbre de décision.

        Paramètres:
        only_leaves : bool, optionnel
            Si True, compte seulement les feuilles (False par défaut).

        Retourne:
        int
            Le nombre de nœuds dans l'arbre.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Retourne une représentation en chaîne de caractères de l'arbre de décision.

        Retourne:
        str
            La représentation en chaîne de l'arbre de décision.
        """
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """
        Retourne la liste de toutes les feuilles dans l'arbre de décision.

        Retourne:
        list
            Une liste de toutes les feuilles dans l'arbre.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Met à jour les bornes pour tous les nœuds dans l'arbre de décision.
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Met à jour la fonction de prédiction pour des prédictions par lots efficaces.
        """
        # Met à jour les bornes pour chaque nœud
        self.update_bounds()

        # Récupère toutes les feuilles
        leaves = self.get_leaves()

        # Met à jour l'indicateur pour chaque feuille et stocke sa contribution
        for leaf in leaves:
            leaf.update_indicator()

        # Définit la fonction de prédiction efficace
        self.predict = lambda A: np.sum(
            [leaf.indicator(A) * leaf.value for leaf in leaves], axis=0
        )

    def pred(self, x):
        """
        Prédit la classe pour un individu unique en utilisant l'arbre de décision.

        Paramètres:
        x : np.ndarray
            Un tableau NumPy 1D représentant les caractéristiques
            d'un individu unique.

        Retourne:
        int
            La classe prédite pour l'individu.
        """
        return self.root.pred(x)

    def random_split_criterion(self, node):
        """
        Détermine un critère de division aléatoire pour un nœud donné.

        Paramètres:
        node : Node
            Le nœud pour lequel le critère de division est déterminé.

        Retourne:
        tuple
            Un tuple contenant l'index de la caractéristique et la valeur seuil.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(self.explanatory
                                                       [:, feature]
                                                       [node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """
        Ajuste l'arbre de décision aux données explicatives et cibles fournies.

        Paramètres:
        explanatory : array-like
            Les variables explicatives.
        target : array-like
            La variable cible.
        verbose : int, optionnel
            Si défini à 1, imprime les détails de l'entraînement
            (par défaut 0).
        """
        # Définit la méthode de critère de division
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
            print(f"""  Entraînement terminé.
        - Profondeur                : {self.depth()}
        - Nombre de nœuds          : {self.count_nodes()}
        - Nombre de feuilles        : {self.count_nodes(only_leaves=True)}""")
            print(f"    - Précision sur les données d'entraînement : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def np_extrema(self, arr):
        """
        Retourne les valeurs minimale et maximale du tableau.

        Paramètres:
        arr : array-like
            Le tableau d'entrée.

        Retourne:
        tuple
            Un tuple contenant les valeurs minimale et maximale du tableau.
        """
        return np.min(arr), np.max(arr)

    def fit_node(self, node):
        """
        Ajuste récursivement les nœuds de l'arbre de décision.

        Paramètres:
        node : Node
            Le nœud actuel en cours d'ajustement.
        """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & \
            (self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & ~left_population
        if len(left_population) != len(self.target):
            left_population = np.pad(
                left_population,
                (0, len(self.target) - len(left_population)),
                'constant', constant_values=(0)
            )
        if len(right_population) != len(self.target):
            right_population = np.pad(
                right_population,
                (0, len(self.target) - len(right_population)),
                'constant', constant_values=(0)
            )
        is_left_leaf = (
            node.depth == self.max_depth - 1 or
            np.sum(left_population) <= self.min_pop or
            np.unique(self.target[left_population]).size == 1
        )
        is_right_leaf = (
            node.depth == self.max_depth - 1 or
            np.sum(right_population) <= self.min_pop or
            np.unique(self.target[right_population]).size == 1
        )
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            node.left_child.depth = node.depth + 1
            self.fit_node(node.left_child)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            node.right_child.depth = node.depth + 1
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Crée un nœud feuille enfant.

        Paramètres:
        node : Node
            Le nœud parent.
        sub_population : array-like
            La sous-population pour le nœud feuille.

        Retourne:
        Leaf
            Le nœud feuille créé.
        """
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Crée un nœud enfant non-feuille.

        Paramètres:
        node : Node
            Le nœud parent.
        sub_population : array-like
            La sous-population pour le nœud enfant.

        Retourne:
        Node
            Le nœud enfant créé.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        Calcule la précision de l'arbre de décision sur les données de test.

        Paramètres:
        test_explanatory : array-like
            Les variables explicatives pour les données de test.
        test_target : array-like
            La variable cible pour les données de test.

        Retourne:
        float
            La précision de l'arbre de décision sur les données de test.
        """
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size
