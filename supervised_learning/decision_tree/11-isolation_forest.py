#!/usr/bin/env python3
""" Task 11: 11. IRF 2 : isolation random forests """
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """
    A class representing an Isolation
    Random Forest, used primarily for outlier detection.

    Attributes:
    numpy_predicts : list
        A list to store predictions from
        individual trees (not used in current implementation).
    target : None
        A placeholder for a target variable,
        which is not used in isolation forests.
    numpy_preds : list of callable
        A list to store the `predict` methods
        from the trained Isolation Random Trees.
    n_trees : int
        The number of trees in the forest.
    max_depth : int
        The maximum depth allowed for each tree.
    seed : int
        The seed for random number generation, used
        to ensure reproducibility.

    Methods:
    __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        Initializes the Isolation_Random_Forest with specified parameters.

    predict(self, explanatory):
        Predicts the likelihood of data points being outliers
        based on the average depth
        across all trees in the forest.

    fit(self, explanatory, n_trees=100, verbose=0):
        Trains the Isolation Random Forest on the given
        explanatory variables.

    suspects(self, explanatory, n_suspects):
        Identifies the data points that are most likely to be outliers
        by selecting those
        with the smallest mean depth across all trees in the forest.
    """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initializes the Isolation_Random_Forest
        instance with the given parameters.

        Parameters:
        n_trees : int, optional
            Number of trees in the forest (default is 100).
        max_depth : int, optional
            Maximum depth of each tree (default is 10).
        min_pop : int, optional
            Minimum population for a node to split (default is 1,
            though not used in the current implementation).
        seed : int, optional
            Random seed for reproducibility (default is 0).
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts outlier scores for the given explanatory
        variables based on the trained forest.

        Parameters:
        explanatory : numpy.ndarray
            The explanatory variables (features) for which
            to predict outlier scores.

        Returns:
        numpy.ndarray
            The average outlier score across all trees for each data point.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
        Trains the Isolation Random Forest on the given
        explanatory variables.

        Parameters:
        explanatory : numpy.ndarray
            The explanatory variables (features) to train the forest on.
        n_trees : int, optional
            Number of trees to build (default is 100).
        verbose : int, optional
            Verbosity mode (0 = silent, 1 = prints training details).

        Returns:
        None
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(
                max_depth=self.max_depth, seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        self.target = self.predict(explanatory)
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory, n_suspects):
        """
        Returns the n_suspects rows in explanatory
        that have the smallest mean depth.

        Parameters:
        explanatory : numpy.ndarray
            The dataset of explanatory variables.
        n_suspects : int
            The number of suspects (rows) to return.

        Returns:
        numpy.ndarray
            The rows of the explanatory dataset
            corresponding to the n_suspects
            with the smallest mean depth.
        """
        depths = self.predict(explanatory)
        suspect_indices = np.argsort(depths)[:n_suspects]
        return explanatory[suspect_indices], depths[suspect_indices]
