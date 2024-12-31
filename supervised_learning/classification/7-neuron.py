#!/usr/bin/env python3
"""
Module définissant un neurone effectuant une classification binaire
avec visualisation de l'entraînement
"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """
    Classe définissant un neurone unique pour la classification binaire
    avec des attributs privés et méthodes d'entraînement avancées
    """

    def __init__(self, nx):
        """
        Initialise un neurone avec des attributs privés

        Args:
            nx (int): nombre de caractéristiques d'entrée du neurone

        Raises:
            TypeError: si nx n'est pas un entier
            ValueError: si nx est inférieur à 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter pour récupérer les poids"""
        return self.__W

    @property
    def b(self):
        """Getter pour récupérer le biais"""
        return self.__b

    @property
    def A(self):
        """Getter pour récupérer la sortie activée"""
        return self.__A

    def forward_prop(self, X):
        """
        Calcule la propagation avant du neurone

        Args:
            X (numpy.ndarray): matrice d'entrée de forme (nx, m)

        Returns:
            float: La sortie activée du neurone
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Calcule le coût du modèle

        Args:
            Y (numpy.ndarray): étiquettes correctes
            A (numpy.ndarray): sorties prédites

        Returns:
            float: Le coût
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Évalue les prédictions du neurone

        Args:
            X (numpy.ndarray): données d'entrée
            Y (numpy.ndarray): étiquettes correctes

        Returns:
            tuple: (prédictions, coût)
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calcule une passe de descente de gradient

        Args:
            X (numpy.ndarray): données d'entrée
            Y (numpy.ndarray): étiquettes correctes
            A (numpy.ndarray): sorties prédites
            alpha (float): taux d'apprentissage
        """
        m = X.shape[1]
        dz = A - Y
        dw = (1 / m) * np.matmul(dz, X.T)
        db = (1 / m) * np.sum(dz)

        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Entraîne le neurone

        Args:
            X (numpy.ndarray): données d'entrée
            Y (numpy.ndarray): étiquettes correctes
            iterations (int): nombre d'itérations
            alpha (float): taux d'apprentissage
            verbose (bool): affiche le coût pendant l'entraînement
            graph (bool): affiche le graphique d'apprentissage
            step (int): pas entre les affichages

        Returns:
            tuple: (prédictions finales, coût final)

        Raises:
            TypeError: si les paramètres ne sont pas du bon type
            ValueError: si les paramètres ne sont pas dans les bonnes plages
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                costs.append(cost)
                steps.append(i)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
