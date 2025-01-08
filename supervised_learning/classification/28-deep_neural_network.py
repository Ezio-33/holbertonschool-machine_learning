#!/usr/bin/env python3
"""
Module 28-deep_neural_network
Réseau de neurones profond pour classification multi-classes
avec choix d'activation (sigmoïde ou tanh).
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """
    Réseau de neurones profond pour classification multi-classes.
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Initialise le réseau.
        Args:
            nx (int): Nombre de caractéristiques en entrée
            layers (list): Nombre de neurones par couche
            activation (str): 'sig' ou 'tanh'
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if any(not isinstance(n, int) or n <= 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")
        if not isinstance(activation, str):
            raise TypeError("activation must be a string")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i, neurons in enumerate(layers):
            layer_num = i + 1
            if i == 0:
                xavier_init = np.sqrt(1 / nx)
                self.__weights['W' + str(layer_num)] = (
                    np.random.randn(neurons, nx) * xavier_init
                )
            else:
                xavier_init = np.sqrt(1 / layers[i - 1])
                self.__weights['W' + str(layer_num)] = (
                    np.random.randn(neurons, layers[i - 1]) * xavier_init
                )
            self.__weights['b' + str(layer_num)] = np.zeros((neurons, 1))

    @property
    def L(self):
        """Nombre total de couches."""
        return self.__L

    @property
    def cache(self):
        """Dictionnaire des activations (A0, A1, ...)."""
        return self.__cache

    @property
    def weights(self):
        """Dictionnaire des poids et biais (W1, b1, ...)."""
        return self.__weights

    @property
    def activation(self):
        """Fonction d'activation choisie ('sig' ou 'tanh')."""
        return self.__activation

    def forward_prop(self, X):
        """
        Calcule la propagation avant.
        - Couches cachées : 'sig' ou 'tanh'
        - Dernière couche : softmax
        Args:
            X (ndarray): Données d'entrée (nx, m)
        Returns:
            tuple: Activation de la dernière couche et cache
        """
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]
            Z = np.matmul(W, A_prev) + b
            if i == self.__L:
                A = self.softmax(Z)
            else:
                if self.__activation == 'tanh':
                    A = self.tanh_activation(Z)
                else:
                    A = self.sigmoid(Z)
            self.__cache['A' + str(i)] = A
        return A, self.__cache

    def sigmoid(self, z):
        """Fonction sigmoïde."""
        return 1 / (1 + np.exp(-z))

    def tanh_activation(self, z):
        """Fonction tanh."""
        return np.tanh(z)

    def softmax(self, z):
        """Fonction softmax."""
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def cost(self, Y, A):
        """Coût cross-entropy pour classification multi-classes."""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A + 1e-15)) / m

    def evaluate(self, X, Y):
        """Évalue les prédictions et calcule le coût."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Descente de gradient (rétropropagation).
        - Couches cachées : dérivée sig/tanh
        - Dernière couche : softmax
        Args:
            Y (ndarray): Étiquettes one-hot encodées (classes, m)
            cache (dict): Dictionnaire des activations
            alpha (float): Taux d'apprentissage
        """
        m = Y.shape[1]
        L = self.__L
        dZ = cache['A' + str(L)] - Y

        for i in range(L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db
            if i > 1:
                W_prev = self.__weights['W' + str(i)]
                dA_prev = np.matmul(W_prev.T, dZ)
                if self.__activation == 'tanh':
                    dZ = dA_prev * (1 - cache['A' + str(i - 1)] ** 2)
                else:
                    A_prev_activ = cache['A' + str(i - 1)]
                    dZ = dA_prev * (A_prev_activ * (1 - A_prev_activ))

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Entraîne le réseau.
        Args:
            X (ndarray): Données d'entrée
            Y (ndarray): Étiquettes one-hot
            iterations (int): Nombre d'itérations
            alpha (float): Taux d'apprentissage
            verbose (bool): Affiche le coût régulièrement
            graph (bool): Trace le coût en fonction des itérations
            step (int): Intervalle d'affichage
        Returns:
            tuple: Prédictions et coût final
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        if not isinstance(graph, bool):
            raise TypeError("graph must be a boolean")
        if (verbose or graph):
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps_list = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost_i = self.cost(Y, A)
            if i % step == 0 or i == iterations:
                costs.append(cost_i)
                steps_list.append(i)
                if verbose:
                    print(f"Coût après {i} itérations: {cost_i}")
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)
        if graph:
            plt.plot(steps_list, costs, 'b-')
            plt.xlabel('itération')
            plt.ylabel('coût')
            plt.title('Coût d\'entraînement')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Sauvegarde l'instance dans un fichier pickle."""
        if not isinstance(filename, str):
            return
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        except Exception:
            return

    @staticmethod
    def load(filename):
        """Charge une instance depuis un fichier pickle."""
        if not isinstance(filename, str):
            return None
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
