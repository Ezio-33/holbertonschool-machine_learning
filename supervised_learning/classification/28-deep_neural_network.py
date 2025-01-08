#!/usr/bin/env python3
"""
module
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Classe DeepNeuralNetwork qui définit un réseau de neurones profond
    réalisant une classification binaire [..]
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Initialise le réseau avec le nombre d'entrées, de couches
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__nx = nx
        self.__layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            wkey = "W{}".format(i + 1)
            bkey = "b{}".format(i + 1)

            self.__weights[bkey] = np.zeros((layers[i], 1))

            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                w = np.random.randn(layers[i], layers[i - 1])
                w = w * np.sqrt(2 / layers[i - 1])
            self.__weights[wkey] = w

    @property
    def L(self):
        """
        Retourne le nombre de couches du réseau.
        """
        return self.__L

    @property
    def cache(self):
        """
        Retourne le cache contenant les valeurs intermédiaires de propagation.
        """
        return self.__cache

    @property
    def weights(self):
        """
        Retourne les poids et biais du réseau.
        """
        return self.__weights

    @property
    def activation(self):
        """
        Retourne la fonction d'activation utilisée dans les couches cachées.
        """
        return self.__activation

    def forward_prop(self, X):
        """
        Effectue la propagation avant à travers le réseau.
        """
        self.__cache['A0'] = X

        for i in range(self.__L):
            wkey = "W{}".format(i + 1)
            bkey = "b{}".format(i + 1)
            Aprevkey = "A{}".format(i)
            Akey = "A{}".format(i + 1)
            W = self.__weights[wkey]
            b = self.__weights[bkey]
            Aprev = self.__cache[Aprevkey]

            z = np.matmul(W, Aprev) + b
            if i < self.__L - 1:
                if self.__activation == 'sig':
                    self.__cache[Akey] = self.sigmoid(z)
                elif self.__activation == 'tanh':
                    self.__cache[Akey] = self.tanh(z)
            else:
                self.__cache[Akey] = self.softmax(z)

        return (self.__cache[Akey], self.__cache)

    def sigmoid(self, z):
        """
        Calcule la fonction sigmoid.
        """
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        """
        Calcule la fonction tanh.
        """
        return np.tanh(z)

    def softmax(self, z):
        """
        Calcule la fonction softmax.
        """
        y_hat = np.exp(z - np.max(z))
        return y_hat / y_hat.sum(axis=0)

    def cost(self, Y, A):
        """
        Calcule le coût de la prédiction.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Évalue les performances du modèle sur des données données.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        Y_hat = np.max(A, axis=0)
        A = np.where(A == Y_hat, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Met à jour les poids et biais du réseau via descente de gradient.
        """
        m = Y.shape[1]
        Al = cache["A{}".format(self.__L)]
        dAl = (-Y / Al) + (1 - Y) / (1 - Al)

        for i in reversed(range(1, self.__L + 1)):
            wkey = "W{}".format(i)
            bkey = "b{}".format(i)
            Al = cache["A{}".format(i)]
            Al1 = cache["A{}".format(i - 1)]

            if i < self.__L:
                if self.__activation == 'sig':
                    g = Al * (1 - Al)
                elif self.__activation == 'tanh':
                    g = 1 - Al ** 2
            else:
                g = Al * (1 - Al)  # Softmax grad simplifiée pour binaire

            dZ = np.multiply(dAl, g)
            dW = np.matmul(dZ, Al1.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            W = self.__weights[wkey]
            dAl = np.matmul(W.T, dZ)

            self.__weights[wkey] = self.__weights[wkey] - alpha * dW
            self.__weights[bkey] = self.__weights[bkey] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Entraîne le réseau de neurones sur un ensemble de données.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and <= iterations")

        cost_list = []
        step_list = []
        for i in range(iterations):
            A, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            cost = self.cost(Y, A)
            if i % step == 0:
                cost_list.append(cost)
                step_list.append(i)
                if verbose:
                    print("Cost after {} iterations: {:.6f}".format(i, cost))

        if graph:
            plt.plot(step_list, cost_list)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Sauvegarde une instance du réseau dans un fichier.
        """
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Charge une instance du réseau depuis un fichier.
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
