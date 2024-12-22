#!/usr/bin/env python3
""" Task 10: 10. IRF 1 : isolation random trees """
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree:
    """
    Isolation_Random_Tree class implements an
    isolation tree for detecting outliers.

    Attributes:
    rng : numpy.random.Generator
        Random number generator initialized with a seed.
    root : Node or Leaf
        Root node of the tree, which can either be a Node or a Leaf.
    explanatory : array-like
        Explanatory variables used for training the tree.
    max_depth : int
        Maximum depth of the tree.
    predict : function
        Function to predict the depth of a given data point.
    min_pop : int
        Minimum population at a node for it to split.

    Methods:
    __init__(self, max_depth=10, seed=0, root=None):
        Initializes the Isolation_Random_Tree with specified parameters.

    __str__(self):
        Returns a string representation of the tree.

    depth(self):
        Returns the depth of the tree.

    count_nodes(self, only_leaves=False):
        Returns the count of nodes in the tree, optionally only leaves.

    update_bounds(self):
        Updates the bounds of the tree.

    get_leaves(self):
        Returns a list of leaves in the tree.

    update_predict(self):
        Updates the predict function of the tree.

    np_extrema(self, arr):
        Returns the minimum and maximum of an array.

    random_split_criterion(self, node):
        Generates a random split criterion for the node.

    get_leaf_child(self, node, sub_population):
        Returns a leaf child node given a parent node and its subpopulation.

    get_node_child(self, node, sub_population):
        Returns a node child given a parent node and its subpopulation.

    fit_node(self, node):
        Recursively fits the node with its children based on random splits.

    fit(self, explanatory, verbose=0):
        Fits the entire tree on the given explanatory data.
    """

    def __init__(self, max_depth=10, seed=0, root=None):
        """
        Initializes the Isolation_Random_Tree with specified parameters.

        Parameters:
        max_depth : int, optional
            Maximum depth of the tree (default is 10).
        seed : int, optional
            Seed for random number generation (default is 0).
        root : Node or Leaf, optional
            Root node of the tree (default is None, which creates a new Node).
        """
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        Returns a string representation of the decision tree.

        Returns:
        str
            The string representation of the decision tree.
        """
        return self.root.__str__() + "\n"

    def depth(self):
        """
        Returns the maximum depth of the tree.

        Returns:
        int
            The maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the number of nodes in the decision tree.

        Parameters:
        only_leaves : bool, optional
            If True, count only the leaf nodes (default is False).

        Returns:
        int
            The number of nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """
        Update the bounds for the entire
        tree starting from the root node.
        """
        self.root.update_bounds_below()

    def get_leaves(self):
        """
        Returns a list of all leaves in the tree.

        Returns:
        list
            The list of all leaves in the tree.
        """
        return self.root.get_leaves_below()

    def update_predict(self):
        """
        Update the prediction function for the decision tree.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict(A):
            """
            Predict the class for each individual in the input
            array A using the decision tree.

            Parameters:
            A : np.ndarray
                A 2D NumPy array of shape (n_individuals,
                n_features), where each row
                represents an individual with its features.

            Returns:
            np.ndarray
                A 1D NumPy array of shape (n_individuals,),
                where each element is the predicted
                class for the corresponding individual in A.
            """
            predictions = np.zeros(A.shape[0], dtype=int)
            for i, x in enumerate(A):
                for leaf in leaves:
                    if leaf.indicator(np.array([x])):
                        predictions[i] = leaf.value
                        break
            return predictions
        self.predict = predict

    def np_extrema(self, arr):
        """
        Returns the minimum and maximum values of an array.

        Parameters:
        arr : array-like
            Array from which to find the extrema.

        Returns:
        tuple
            Minimum and maximum values of the array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Determines a random split criterion for a given node.

        Parameters
        node : Node
            The node for which the split criterion is determined.

        Returns
        tuple
            A tuple containing the feature index and the threshold value.
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

    def get_leaf_child(self, node, sub_population):
        """
        Returns a leaf child node given a parent node and its subpopulation.

        Parameters:
        node : Node
            The parent node.
        sub_population : array-like
            Subpopulation of the explanatory data for the child node.

        Returns:
        Leaf
            A leaf child node with the updated depth and subpopulation.
        """
        value = node.depth + 1
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Creates a non-leaf child node.

        Parameters
        node : Node
            The parent node.
        sub_population : array-like
            The sub-population for the child node.

        Returns
        Node
            The created non-leaf child node.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        Recursively fits the node with its children based on random splits.

        Parameters:
        node : Node
            The node to fit.
        """

        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & \
            (self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & ~left_population

        is_left_leaf = (node.depth == self.max_depth - 1 or
                        np.sum(left_population) <= self.min_pop)
        is_right_leaf = (node.depth == self.max_depth - 1 or
                         np.sum(right_population) <= self.min_pop)

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

    def fit(self, explanatory, verbose=0):
        """
        Fits the entire Isolation_Random_Tree on the given explanatory data.

        Parameters:
        explanatory : array-like
            Explanatory variables used for training.
        verbose : int, optional
            If set to 1, prints training statistics (default is 0).
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
