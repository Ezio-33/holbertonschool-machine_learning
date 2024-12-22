#!/usr/bin/env python3
"""Task 9: 9. Random forests"""
import numpy as np
from scipy import stats
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest:
    """
    Random_Forest class implements a random forest algorithm which
    builds a large list of decision trees with random splitting criteria.

    Attributes:
    n_trees : int
        Number of trees in the forest.
    max_depth : int
        Maximum depth of the trees.
    min_pop : int
        Minimum population at a node for it to split.
    seed : int
        Seed for random number generation.
    numpy_preds : list
        List of prediction functions from each tree.
    target : array-like
        Target variable used during training.
    explanatory : array-like
        Explanatory variables used during training.

    Methods:
    __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        Initializes the Random_Forest with the specified parameters.

    predict(self, explanatory):
        Predicts the class labels for the given explanatory
        data based on the majority vote of all trees.

    fit(self, explanatory, target, n_trees=100, verbose=0):
        Trains the Random_Forest on the given explanatory and
        target data by building decision trees.

    accuracy(self, test_explanatory, test_target):
        Calculates the accuracy of the Random_Forest on test data.
    """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initializes the Random_Forest with specified parameters.

        Parameters:
        n_trees : int, optional
            Number of trees in the forest (default is 100).
        max_depth : int, optional
            Maximum depth of the trees (default is 10).
        min_pop : int, optional
            Minimum population at a node for it to split (default is 1).
        seed : int, optional
            Seed for random number generation (default is 0).
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts the class labels for the given explanatory data.

        Parameters:
        explanatory : array-like
            Explanatory variables for which predictions are required.

        Returns:
        array-like
            Predicted class labels.
        """
        all_preds = []
        for tree_predict in self.numpy_preds:
            preds = tree_predict(explanatory)
            all_preds.append(preds)
        all_preds = np.array(all_preds)
        mode_preds = stats.mode(all_preds, axis=0)[0]
        return mode_preds.flatten()

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """
        Trains the Random_Forest on the given explanatory and target data.

        Parameters:
        explanatory : array-like
            Explanatory variables used for training.
        target : array-like
            Target variable used for training.
        n_trees : int, optional
            Number of trees in the forest (default is 100).
        verbose : int, optional
            If set to 1, prints training statistics (default is 0).
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop, seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}""")
            print(f"    - Accuracy of the forest on td   : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def accuracy(self, test_explanatory, test_target):
        """
        Calculates the accuracy of the Random_Forest on test data.

        Accuracy is calculated as the proportion of correctly predicted
        labels out of the total labels.

        Parameters:
        test_explanatory : array-like
            Explanatory variables of the test data.
        test_target : array-like
            True target labels of the test data.

        Returns:
        float
            Accuracy of the Random_Forest on the test data.
        """

        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size
