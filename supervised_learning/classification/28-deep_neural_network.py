#!/usr/bin/env python3
"""
Module 28-deep_neural_network
Réseau de neurones profond pour classification multiclasse
avec choix d'activation (sigmoïde ou tanh).
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """
    Classe DeepNeuralNetwork qui définit un réseau de neurones profond
    réalisant une classification multiclasse.
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Initialise le réseau.

        Args:
            nx (int): Nombre de caractéristiques en entrée
            layers (list): Nombre de neurones par couche
            activation (str): 'sig' ou 'tanh'

        Raises:
            TypeError: Si les paramètres ne sont pas du bon type.
            ValueError: Si les paramètres ont des valeurs inappropriées.
        """
        if not isinstance(nx, int):
            raise TypeError("nx doit être un entier")
        if nx < 1:
            raise ValueError("nx doit être un entier positif")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers doit être une liste d'entiers positifs")
        if any(not isinstance(n, int) or n <= 0 for n in layers):
            raise TypeError("layers doit être une liste d'entiers positifs")
        if not isinstance(activation, str):
            raise TypeError("activation doit être une chaîne de caractères")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation doit être 'sig' ou 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        # Initialisation He
        for i, neurons in enumerate(layers):
            layer_num = i + 1
            if i == 0:
                he_init = np.sqrt(2 / nx)
                self.__weights['W' + str(layer_num)] = (
                    np.random.randn(neurons, nx) * he_init
                )
            else:
                he_init = np.sqrt(2 / layers[i - 1])
                self.__weights['W' + str(layer_num)] = (
                    np.random.randn(neurons, layers[i - 1]) * he_init
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
                expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = expZ / np.sum(expZ, axis=0, keepdims=True)
            else:
                if self.__activation == 'tanh':
                    A = np.tanh(Z)
                else:
                    A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(i)] = A
        return A, self.__cache

    def sigmoid(self, z):
        """Calcule la fonction sigmoïde."""
        return 1 / (1 + np.exp(-z))

    def tanh_activation(self, z):
        """Calcule la fonction tanh."""
        return np.tanh(z)

    def cost(self, Y, A):
        """
        Calcule le coût cross-entropy pour classification multiclasse.

        Args:
            Y (ndarray): Étiquettes one-hot encodées (classes, m)
            A (ndarray): Prédictions du modèle (classes, m)

        Returns:
            float: Coût moyen
        """
        m = Y.shape[1]
        return -np.sum(Y * np.log(A)) / m

    def evaluate(self, X, Y):
        """
        Évalue les prédictions et calcule le coût.

        Args:
            X (ndarray): Données d'entrée
            Y (ndarray): Étiquettes

        Returns:
            tuple: Prédictions one-hot et coût
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.argmax(A, axis=0)
        A_pred = np.zeros_like(A)
        A_pred[predictions, np.arange(A.shape[1])] = 1
        return A_pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Descente de gradient (rétropropagation).

        Args:
            Y (ndarray): Étiquettes one-hot encodées (classes, m)
            cache (dict): Dictionnaire des activations
            alpha (float): Taux d'apprentissage
        """
        m = Y.shape[1]
        L = self.__L
        dZ = cache['A' + str(L)] - Y  # Softmax derivative for multiclass

        for i in range(L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            W = self.__weights['W' + str(i)]
            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db

            if i > 1:
                W_prev = self.__weights['W' + str(i)]
                A_prev = cache['A' + str(i - 1)]
                if self.__activation == 'tanh':
                    dA_prev = np.matmul(W_prev.T, dZ)
                    dZ = dA_prev * (1 - A_prev ** 2)
                else:
                    dA_prev = np.matmul(W_prev.T, dZ)
                    dZ = dA_prev * (A_prev * (1 - A_prev))

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Entraîne le réseau de neurones.

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
            raise TypeError("iterations doit être un entier")
        if iterations <= 0:
            raise ValueError("iterations doit être un entier positif")
        if not isinstance(alpha, float):
            raise TypeError("alpha doit être un float")
        if alpha <= 0:
            raise ValueError("alpha doit être positif")
        if not isinstance(verbose, bool):
            raise TypeError("verbose doit être un booléen")
        if not isinstance(graph, bool):
            raise TypeError("graph doit être un booléen")
        if (verbose or graph):
            if not isinstance(step, int):
                raise TypeError("step doit être un entier")
            if step <= 0 or step > iterations:
                raise ValueError("step doit être positif et <= iterations")

        costs = []
        steps_list = []

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            cost = self.cost(Y, A)
            if i % step == 0:
                costs.append(cost)
                steps_list.append(i)
                if verbose:
                    print(f"Coût après {i} itérations: {cost}")

        if graph:
            plt.plot(steps_list, costs, 'b-')
            plt.xlabel('itération')
            plt.ylabel('coût')
            plt.title("Coût d'entraînement")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Sauvegarde l'instance dans un fichier pickle.

        Args:
            filename (str): Nom du fichier

        Raises:
            TypeError: Si filename n'est pas une chaîne de caractères
            ValueError: Si la sauvegarde échoue
        """
        if not isinstance(filename, str):
            raise TypeError("filename doit être une chaîne de caractères")
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        except Exception:
            raise ValueError("Impossible de sauvegarder l'objet")

    @staticmethod
    def load(filename):
        """
        Charge une instance depuis un fichier pickle.

        Args:
            filename (str): Nom du fichier

        Returns:
            DeepNeuralNetwork: Instance chargée ou None en cas d'erreur
        """
        if not isinstance(filename, str):
            return None
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
