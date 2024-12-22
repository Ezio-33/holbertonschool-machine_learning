#!/usr/bin/env python3
""" Tâche 8: """
import numpy as np


def left_child_add_prefix(text):
    """
    Ajoute un préfixe à chaque ligne du texte pour
    indiquer qu'il s'agit du fils gauche dans la structure de l'arbre.

    Paramètres:
    text : str
        Le texte auquel le préfixe sera ajouté.

    Retourne:
    str
        Le texte avec le préfixe du fils gauche ajouté à chaque ligne.
    """
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("    |  " + x) + "\n"
    return new_text


def right_child_add_prefix(text):
    """
    Ajoute un préfixe à chaque ligne du texte pour indiquer
    qu'il s'agit du fils droit dans la structure de l'arbre.

    Paramètres:
    text : str
        Le texte auquel le préfixe sera ajouté.

    Retourne:
    str
        Le texte avec le préfixe du fils droit ajouté à chaque ligne.
    """
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("       " + x) + "\n"
    return new_text


class Node:
    """
    Une classe représentant un nœud dans un arbre de décision.

    Attributs:
    feature : int ou None
        La caractéristique utilisée pour diviser les données.
    threshold : float ou None
        La valeur seuil pour la division.
    left_child : Node ou None
        Le nœud enfant gauche.
    right_child : Node ou None
        Le nœud enfant droit.
    is_leaf : bool
        Indique si le nœud est une feuille.
    is_root : bool
        Indique si le nœud est la racine.
    sub_population : any
        La sous-population de données à ce nœud.
    depth : int
        La profondeur du nœud dans l'arbre.

    Méthodes:
    max_depth_below():
        Calcule la profondeur maximale du sous-arbre enraciné à ce nœud.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """
        Initialise un nœud avec les paramètres donnés.

        Paramètres:
        feature : int ou None, optionnel
            La caractéristique utilisée pour diviser les données
            (par défaut None).
        threshold : float ou None, optionnel
            La valeur seuil pour la division (par défaut None).
        left_child : Node ou None, optionnel
            Le nœud enfant gauche (par défaut None).
        right_child : Node ou None, optionnel
            Le nœud enfant droit (par défaut None).
        is_root : bool, optionnel
            Indique si le nœud est la racine (par défaut False).
        depth : int, optionnel
            La profondeur du nœud dans l'arbre (par défaut 0).
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

        Retourne:
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
        Compte le nombre de nœuds dans le sous-arbre enraciné
        à ce nœud.

        Paramètres:
        only_leaves : bool, optionnel
            Si True, compte uniquement les nœuds feuilles
            (par défaut False).

        Retourne:
        int
            Le nombre de nœuds dans le sous-arbre.
        """
        if self.is_leaf:
            return 1
        if self.left_child:
            left_count = self.left_child.count_nodes_below(only_leaves)
        else:
            left_count = 0
        if self.right_child:
            right_count = self.right_child.count_nodes_below(only_leaves)
        else:
            right_count = 0
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def __str__(self):
        """
        Retourne une représentation sous forme de chaîne
        du nœud et de ses enfants.

        Retourne:
        str
            La représentation sous forme de chaîne du nœud.
        """
        if self.is_root:
            Type = "root "
        elif self.is_leaf:
            return f"-> leaf [value={self.value}]"
        else:
            Type = "-> node "
        if self.left_child:
            left_str = left_child_add_prefix(str(self.left_child))
        else:
            left_str = ""
        if self.right_child:
            right_str = right_child_add_prefix(str(self.right_child))
        else:
            right_str = ""
        return f"{Type}[feature={self.feature}, threshold=\
{self.threshold}]\n{left_str}{right_str}".rstrip()

    def get_leaves_below(self):
        """
        Retourne une liste de toutes les feuilles en dessous
        de ce nœud.

        Retourne:
        list
            La liste de toutes les feuilles en dessous
            de ce nœud.
        """
        if self.is_leaf:
            return [self]
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Met à jour les limites pour le nœud actuel et
        propage les limites à ses enfants.
        """
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """
        Calcule la fonction indicatrice pour le nœud actuel
        basé sur les limites inférieure et supérieure.
        """

        def is_large_enough(x):
            """
            Vérifie si chaque individu a toutes ses
            caractéristiques supérieures aux limites inférieures.

            Paramètres:
            x : np.ndarray
                Un tableau NumPy 2D de forme
                (n_individus, n_caractéristiques).

            Retourne:

            np.ndarray
                Un tableau NumPy 1D de valeurs booléennes
                indiquant si chaque individu répond
                à la condition.
            """
            return np.all(np.array([x[:, key] > self.lower[key]
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(x):
            """
            Vérifie si chaque individu a toutes ses
            caractéristiques inférieures ou égales aux limites supérieures.

            Paramètres:
            x : np.ndarray
                Un tableau NumPy 2D de forme
                (n_individus, n_caractéristiques).

            Retourne:
            np.ndarray
                Un tableau NumPy 1D de valeurs booléennes indiquant
                si chaque individu répond
                à la condition.
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
            Un tableau NumPy 1D représentant les
            caractéristiques d'un individu unique.

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
    Une classe représentant un nœud feuille dans un arbre de décision,
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
            La profondeur de la feuille dans l'arbre
            (par défaut None).
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
        Compte le nombre de nœuds dans le sous-arbre enraciné
        à cette feuille.

        Paramètres:
        only_leaves : bool, optionnel
            Si True, compte uniquement les nœuds feuilles
            (par défaut False).

        Retourne:
        int
            Le nombre de nœuds dans le sous-arbre.
        """
        return 1

    def __str__(self):
        """
        Retourne une représentation sous forme de chaîne
        du nœud feuille.

        Retourne:
        str
            La représentation sous forme de chaîne du nœud feuille.
        """
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """
        Retourne une liste de toutes les feuilles en dessous
        de cette feuille.

        Retourne:
        list
            La liste de toutes les feuilles en dessous
            de cette feuille.
        """
        return [self]

    def update_bounds_below(self):
        """
        Fonction de remplacement pour mettre à jour les
        limites pour le nœud actuel et propager les limites
        à ses enfants.
        """
        pass

    def pred(self, x):
        """
        Prédit la classe pour un individu unique au nœud feuille.

        Paramètres:
        x : np.ndarray
            Un tableau NumPy 1D représentant les
            caractéristiques d'un individu unique.

        Retourne:
        int
            La classe prédite pour l'individu.
        """
        return self.value


class Decision_Tree():
    """
    Une classe représentant un arbre de décision.

    Attributs:
    rng : numpy.random.Generator
        Générateur de nombres aléatoires pour la reproductibilité.
    root : Node
        Le nœud racine de l'arbre.
    explanatory : any
        Les caractéristiques explicatives du dataset.
    target : any
        Les valeurs cibles du dataset.
    max_depth : int
        La profondeur maximale de l'arbre.
    min_pop : int
        La population minimale requise pour diviser un nœud.
    split_criterion : str
        Le critère utilisé pour diviser les nœuds.
    predict : any
        Méthode pour prédire la valeur cible pour un ensemble de
        caractéristiques.

    Méthodes:
    depth():
        Retourne la profondeur maximale de l'arbre.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialise un Decision_Tree avec les paramètres donnés.

        Paramètres:
        max_depth : int, optionnel
            La profondeur maximale de l'arbre
            (par défaut 10).
        min_pop : int, optionnel
            La population minimale requise pour diviser un nœud
            (par défaut 1).
        seed : int, optionnel
            Graine pour le générateur de nombres aléatoires
            (par défaut 0).
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
            Si True, compte uniquement les nœuds feuilles
            (par défaut False).

        Retourne:
        int
            Le nombre de nœuds dans l'arbre.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Retourne une représentation sous forme de chaîne
        de l'arbre de décision.

        Retourne:
        str
            La représentation sous forme de chaîne de l'arbre.
        """
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """
        Retourne une liste de toutes les feuilles de l'arbre.

        Retourne:
        list
            La liste de toutes les feuilles de l'arbre.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Met à jour les limites pour l'ensemble de l'arbre
        en partant du nœud racine.
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
            """
            Prédit la classe pour chaque individu dans le tableau
            d'entrée A en utilisant l'arbre de décision.

            Paramètres:
            A : np.ndarray
                Un tableau NumPy 2D de forme (n_individus,
                n_caractéristiques), où chaque ligne
                représente un individu avec ses caractéristiques.

            Retourne:
            np.ndarray
                Un tableau NumPy 1D de forme (n_individus,),
                où chaque élément est la classe prédite
                pour l'individu correspondant dans A.
            """
            predictions = np.zeros(A.shape[0], dtype=int)
            for i, x in enumerate(A):
                for leaf in leaves:
                    if leaf.indicator(np.array([x])):
                        predictions[i] = leaf.value
                        break
            return predictions
        self.predict = predict

    def pred(self, x):
        """
        Prédit la classe pour un individu unique en utilisant
        l'arbre de décision.

        Paramètres:
        x : np.ndarray
            Un tableau NumPy 1D représentant les
            caractéristiques d'un individu unique.

        Retourne:
        int
            La classe prédite pour l'individu.
        """
        return self.root.pred(x)

    def np_extrema(self, arr):
        """
        Retourne les valeurs minimale et maximale du tableau.

        Paramètres:
        arr : array-like
            Le tableau d'entrée.

        Retourne:
        tuple
            Un tuple contenant les valeurs minimale et
            maximale du tableau.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Détermine un critère de division aléatoire pour un
        nœud donné.

        Paramètres:
        node : Node
            Le nœud pour lequel le critère de division
            est déterminé.

        Retourne:
        tuple
            Un tuple contenant l'indice de la caractéristique
            et la valeur seuil.
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
        Ajuste l'arbre de décision aux données explicatives et cibles
        fournies.

        Paramètres:
        explanatory : array-like
            Les variables explicatives.
        target : array-like
            La variable cible.
        verbose : int, optionnel
            Si défini à 1, affiche les détails de l'entraînement
            (par défaut 0).
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
            print(f"""  Entraînement terminé.
        - Profondeur                : {self.depth()}
        - Nombre de nœuds          : {self.count_nodes()}
        - Nombre de feuilles        : {self.count_nodes(only_leaves=True)}""")
            print(f"    - Précision sur les données d'entraînement : "
                  f"{self.accuracy(self.explanatory, self.target)}")

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
            left_population = np.pad(left_population,
                                     (0, len(self.target) -
                                      len(self.left_population)),
                                     'constant', constant_values=(0))
        if len(right_population) != len(self.target):
            right_population = np.pad(right_population,
                                      (0, len(self.target) -
                                       len(self.right_population)),
                                      'constant', constant_values=(0))
        is_left_leaf = (node.depth == self.max_depth - 1 or
                        np.sum(left_population) <= self.min_pop or
                        np.unique(self.target[left_population]).size == 1)
        is_right_leaf = (node.depth == self.max_depth - 1 or
                         np.sum(right_population) <= self.min_pop or
                         np.unique(self.target[right_population]).size == 1)
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
        Crée un nœud enfant feuille.

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
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size

    def possible_thresholds(self, node, feature):
        """
        Calcule les seuils possibles pour diviser un nœud de l'arbre
        de décision basé sur une caractéristique spécifique.

        Paramètres:
        node : Node
            Le nœud de l'arbre de décision pour lequel
            les seuils doivent être calculés.
        feature : int
            L'indice de la caractéristique (colonne)
            dans les variables explicatives (caractéristiques)
            du dataset.

        Retourne:
        numpy.ndarray
            Un tableau 1D contenant les seuils possibles
            pour diviser le nœud.
        """
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """
        Calcule l'impureté de Gini pour tous les seuils
        possibles d'une caractéristique donnée et retourne
        le seuil qui minimise l'impureté de Gini
        ainsi que la valeur d'impureté correspondante.

        Paramètres:
        node : Node
            Le nœud de l'arbre de décision pour lequel
            l'impureté de Gini doit être calculée.
        feature : int
            L'indice de la caractéristique à évaluer.

        Retourne:
        numpy.ndarray
            Un tableau 1D contenant le meilleur seuil et
            la valeur minimale de l'impureté de Gini.
        """
        thresholds = self.possible_thresholds(node, feature)
        indices = np.arange(self.explanatory.shape[0])[node.sub_population]
        feature_values = self.explanatory[indices, feature]
        target_reduced = self.target[indices]
        classes = np.unique(target_reduced)

        gini_sum = []
        for threshold in thresholds:
            left_indices = feature_values > threshold
            right_indices = ~left_indices

            gini_left, gini_right = 0, 0
            for a in classes:
                p_left = np.mean(target_reduced[left_indices] == a)
                p_right = np.mean(target_reduced[right_indices] == a)
                gini_left += p_left * (1 - p_left)
                gini_right += p_right * (1 - p_right)

            left_size = np.sum(left_indices)
            right_size = np.sum(right_indices)
            total_size = left_size + right_size
            gini_sum1 = ((gini_left * left_size + gini_right * right_size)
                         / total_size)
            gini_sum.append(gini_sum1)

        min_index = np.argmin(gini_sum)
        return np.array([thresholds[min_index], gini_sum[min_index]])

    def Gini_split_criterion(self, node):
        """
        Détermine la meilleure caractéristique et son
        impureté de Gini associée pour diviser un
        nœud de l'arbre de décision.

        Paramètres:
        node : Node
            Le nœud de l'arbre de décision pour lequel
            le critère de division de Gini
            doit être calculé.

        Retourne:
        tuple (int, float)
            Un tuple où:
            - Le premier élément est l'indice de la caractéristique
              qui aboutit à la meilleure division (impureté
              de Gini la plus basse).
            - Le deuxième élément est la valeur de l'impureté de Gini
              associée à cette meilleure division.
        """
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]
