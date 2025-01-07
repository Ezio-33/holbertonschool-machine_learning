#!/usr/bin/env python3
"""
Module 28-deep_neural_network
Définit un réseau de neurones profond réalisant une classification multi-classes,
avec la possibilité de choisir l'activation (sigmoid ou tanh) pour les couches cachées.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Classe DeepNeuralNetwork
    Implémente un réseau de neurones profond performant
    pour la classification multi-classes, avec différentes activations.
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Initialise un réseau de neurones profond.

        Paramètres:
        -----------
        nx : int
            nombre de caractéristiques en entrée
        layers : list
            liste contenant le nombre de neurones par couche
        activation : str
            'sig' (sigmoïde par défaut) ou 'tanh' (hyperbolique)
            pour les couches cachées

        Exceptions:
        -----------
        TypeError :
            - si nx n'est pas un entier
            - si layers n'est pas une liste
            - si un élément de layers n'est pas un entier positif
        ValueError :
            - si nx < 1
            - si layers est vide
            - si activation n'est ni 'sig' ni 'tanh'
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if any(not isinstance(n, int) or n <= 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        # Initialisation He
        for i, nb_neurons in enumerate(layers):
            if i == 0:
                he_init = np.sqrt(2 / nx)
                W = np.random.randn(nb_neurons, nx) * he_init
            else:
                he_init = np.sqrt(2 / layers[i - 1])
                W = np.random.randn(nb_neurons, layers[i - 1]) * he_init

            b = np.zeros((nb_neurons, 1))
            self.__weights['W' + str(i + 1)] = W
            self.__weights['b' + str(i + 1)] = b

    @property
    def L(self):
        """Getter : nombre total de couches"""
        return self.__L

    @property
    def cache(self):
        """Getter : Dictionnaire qui stocke les activations (A0, A1, ...)"""
        return self.__cache

    @property
    def weights(self):
        """Getter : Dictionnaire qui stocke les poids et biais (W1, b1, ...)"""
        return self.__weights

    @property
    def activation(self):
        """Getter : indique l'activation ('sig' ou 'tanh') pour les couches cachées"""
        return self.__activation

    def sigmoid(self, Z):
        """
        Applique la fonction d'activation sigmoïde.

        Args:
            Z (numpy.ndarray): Valeurs d'entrée.

        Returns:
            numpy.ndarray: Sortie après sigmoïde.
        """
        return 1 / (1 + np.exp(-Z))

    def softmax(self, Z):
        """
        Applique la fonction d'activation softmax.

        Args:
            Z (numpy.ndarray): Valeurs d'entrée.

        Returns:
            numpy.ndarray: Sortie après softmax.
        """
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward_prop(self, X):
        """
        Calcule la propagation avant du réseau.

        - Les couches cachées utilisent l'activation choisie (sigmoid ou tanh).
        - La couche de sortie utilise softmax (multi-classes).

        Paramètres:
        -----------
        X : ndarray (nx, m)
            Données d'entrée

        Returns:
        --------
        A : ndarray
            Activation de la dernière couche
        cache : dict
            Dictionnaire contenant toutes les activations intermédiaires.
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
                    A = np.tanh(Z)
                else:  # 'sig'
                    A = self.sigmoid(Z)

            self.__cache['A' + str(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calcule le coût (cross-entropy catégorique) pour la classification multi-classes.

        Paramètres:
        -----------
        Y : ndarray (classes, m)
            étiquettes one-hot encodées
        A : ndarray (classes, m)
            prédictions après softmax

        Returns:
        --------
        float
            Coût moyen.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-15)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Évalue le réseau de neurones pour la classification multi-classes.

        Paramètres:
        -----------
        X : ndarray (nx, m)
            données d'entrée
        Y : ndarray (classes, m)
            étiquettes one-hot encodées

        Returns:
        --------
        tuple
            (prédictions, coût, exactitude)
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.argmax(A, axis=0).reshape(1, -1)
        labels = np.argmax(Y, axis=0).reshape(1, -1)
        accuracy = np.sum(predictions == labels) / Y.shape[1] * 100
        return predictions, cost, accuracy

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calcule la descente de gradient (rétropropagation).

        - Couches cachées : activation 'sig' => dérivée sigmoïde
                           activation 'tanh' => dérivée tanh
        - Dernière couche : softmax

        Paramètres:
        -----------
        Y : ndarray (classes, m)
            étiquettes one-hot encodées
        cache : dict
            contient les activations A0, A1, ...
        alpha : float
            taux d'apprentissage
        """
        m = Y.shape[1]
        L = self.__L

        # Initialisation du gradient pour la sortie (softmax)
        A_last = cache['A' + str(L)]
        dZ = A_last - Y  # (classes, m)

        for i in range(L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            W = self.__weights['W' + str(i)]
            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            # Mise à jour des poids et biais
            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db

            if i > 1:
                A_prev_activ = cache['A' + str(i - 1)]
                if self.__activation == 'tanh':
                    # Dérivée tanh = 1 - A^2
                    dA_prev = np.matmul(W.T, dZ)
                    dZ = dA_prev * (1 - A_prev_activ**2)
                else:
                    # Dérivée sigmoïde = A * (1 - A)
                    dA_prev = np.matmul(W.T, dZ)
                    dZ = dA_prev * (A_prev_activ * (1 - A_prev_activ))

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Entraîne le réseau de neurones profond.

        Paramètres:
        -----------
        X : ndarray (nx, m)
            données d'entrée
        Y : ndarray (classes, m)
            étiquettes
        iterations : int
            nombre d'itérations
        alpha : float
            taux d'apprentissage
        verbose : bool
            affiche le coût à intervalles réguliers si True
        graph : bool
            trace l'évolution du coût si True
        step : int
            intervalle d'affichage

        Returns:
        --------
        tuple
            (prédictions, coût, exactitude)
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if (verbose or graph):
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost_i = self.cost(Y, A)

            if i % step == 0 or i == iterations:
                costs.append(cost_i)
                steps.append(i)
                if verbose:
                    print(f"Cost after {i} iterations: {cost_i}")

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Sauvegarde l'instance dans un fichier pickle.

        Paramètres:
        -----------
        filename : str
            nom du fichier, on ajoute .pkl si absent
        """
        if not isinstance(filename, str):
            return None
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        except Exception:
            return None

    @staticmethod
    def load(filename):
        """
        Charge une instance sauvegardée via pickle.

        Paramètres:
        -----------
        filename : str
            nom du fichier pickle

        Returns:
        --------
        DeepNeuralNetwork ou None si erreur
        """
        if not isinstance(filename, str):
            return None
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
