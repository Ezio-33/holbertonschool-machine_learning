#!/usr/bin/env python3
"""
Module définissant un réseau de neurones profond
pour la classification binaire.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Classe DeepNeuralNetwork qui définit un réseau de neurones profond
    réalisant une classification binaire
    """

    def __init__(self, nx, layers):
        """
        Constructeur de la classe
        Args :
         - nx (int) : nombre de caractéristiques d'entrée du neurone
         - layers (list) : représentant le nombre de
         nœuds dans chaque couche du réseau
         Attributs d'instance publics :
         - L : Nombre de couches dans le réseau de neurones.
         - cache : Un dictionnaire pour contenir toutes
         les valeurs intermédiaires du réseau.
         - weights : Un dictionnaire pour contenir tous
         les poids et biais du réseau.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__nx = nx
        self.__layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
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
        Fonction getter pour L
        Returns le nombre de couches
        """
        return self.__L

    @property
    def cache(self):
        """
        Fonction getter pour cache
        Returns un dictionnaire pour contenir toutes les
        valeurs intermédiaires
        du réseau
        """
        return self.__cache

    @property
    def weights(self):
        """
        Fonction getter pour weights
        Returns un dictionnaire pour contenir tous les poids et biais du
        réseau
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calcule la propagation avant du réseau de neurones
        Args :
         - X (numpy.ndarray) : avec forme (nx, m) contenant
        les données d'entrée
           * nx est le nombre de caractéristiques d'entrée du neurone
           * m est le nombre d'exemples
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
                self.__cache[Akey] = self.sigmoid(z)
            else:
                self.__cache[Akey] = self.softmax(z)

        return (self.__cache[Akey], self.__cache)

    def sigmoid(self, z):
        """
        Applique la fonction d'activation sigmoïde
        Args :
        - z (numpy.ndarray) : avec forme (nx, m) contenant
        les données d'entrée
         * nx est le nombre de caractéristiques d'entrée du neurone.
         * m est le nombre d'exemples
        Mets à jour l'attribut privé __A
        Le neurone doit utiliser une fonction d'activation sigmoïde
        Returns :
        L'attribut privé A
        """
        y_hat = 1 / (1 + np.exp(-z))
        return y_hat

    def softmax(self, z):
        """
        Applique la fonction d'activation softmax
        Args :
        - z (numpy.ndarray) : avec forme (nx, m) contenant les données d'entrée
         * nx est le nombre de caractéristiques d'entrée du neurone.
         * m est le nombre d'exemples
        Mets à jour l'attribut privé __A
        Le neurone doit utiliser une fonction d'activation softmax

        Returns :
        L'attribut privé y_hat
        """
        y_hat = np.exp(z - np.max(z))
        return y_hat / y_hat.sum(axis=0)

    def cost(self, Y, A):
        """
        Calcule le coût du modèle en utilisant la régression logistique
        Args :
         - Y (numpy.ndarray) : avec forme (1, m) contenant
         les étiquettes correctes pour les données d'entrée
         - A (numpy.ndarray) : avec forme (1, m) contenant la sortie activée
         du neurone pour chaque exemple
        Returns :
         Le coût
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m

        return cost

    def evaluate(self, X, Y):
        """
        Évalue les prédictions du réseau de neurones
        Args :
         - X est un numpy.ndarray avec forme (nx, m) contenant les données
           d'entrée
           * nx est le nombre de caractéristiques d'entrée du neurone
           * m est le nombre d'exemples
         - Y (numpy.ndarray) : avec forme (1, m) contenant les
         étiquettes correctes pour les données d'entrée
        Returns :
         La prédiction du neurone et le coût du réseau, respectivement
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        Y_hat = np.max(A, axis=0)
        A = np.where(A == Y_hat, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Effectue une passe de descente de gradient sur le réseau de neurones
        Args :
         - Y (numpy.ndarray) avec forme (1, m) contenant les
         étiquettes correctes pour les données d'entrée
         - cache (dictionary) : contenant toutes les valeurs intermédiaires du
           réseau
         - alpha (float) : le taux d'apprentissage
        """
        m = Y.shape[1]
        Al = cache["A{}".format(self.__L)]
        dAl = (-Y / Al) + (1 - Y)/(1 - Al)

        for i in reversed(range(1, self.__L + 1)):
            wkey = "W{}".format(i)
            bkey = "b{}".format(i)
            Al = cache["A{}".format(i)]
            Al1 = cache["A{}".format(i - 1)]
            g = Al * (1 - Al)
            dZ = np.multiply(dAl, g)
            dW = np.matmul(dZ, Al1.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            W = self.__weights["W{}".format(i)]
            dAl = np.matmul(W.T, dZ)

            self.__weights[wkey] = self.__weights[wkey] - alpha * dW
            self.__weights[bkey] = self.__weights[bkey] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Entraîne le réseau de neurones profond en mettant à jour les attributs
        privés
        Args :
         - X (numpy.ndarray) : avec forme (nx, m) contenant
         les données d'entrée
           * nx est le nombre de caractéristiques d'entrée du neurone
           * m est le nombre d'exemples
         - Y (numpy.ndarray) : avec forme (1, m) contenant
         les étiquettes correctes pour les données d'entrée
         - iterations (int) : nombre d'itérations pour l'entraînement
         - alpha (float) : taux d'apprentissage
         - verbose (booléen) : définit si oui ou non afficher
              des informations sur l'entraînement
         - graph (booléen) : définit si oui ou non tracer des informations
              sur l'entraînement une fois terminé
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
            cost_list.append(cost)
            step_list.append(i)
            if verbose and i % step == 0:
                print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(step_list, cost_list)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Trainig Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Sauvegarde l'objet instance dans un fichier au format pickle

        Args :
        - filename est le fichier dans lequel l'objet doit être sauvegardé
        Si filename n'a pas l'extension .pkl, l'ajouter

        """
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Charge un objet DeepNeuralNetwork picklé

        Args :
        - filename est le fichier à partir duquel l'objet doit être chargé

        Returns :
        L'objet chargé, ou None si filename n'existe pas
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
