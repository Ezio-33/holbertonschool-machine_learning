#!/usr/bin/env python3
"""
Module définissant un réseau de neurones avec une couche cachée
pour la classification binaire
"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    Classe définissant un réseau de neurones avec une couche cachée
    et des attributs privés pour la classification binaire
    """
    def __init__(self, nx, nodes):
        """
        Initialise le réseau de neurones

        Args:
            nx (int): nombre de caractéristiques d'entrée
            nodes (int): nombre de nœuds dans la couche cachée

        Raises:
            TypeError: si nx ou nodes n'est pas un entier
            ValueError: si nx ou nodes est inférieur à 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter pour W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter pour b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter pour A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter pour W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter pour b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter pour A2"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calcule la propagation avant du réseau de neurones

        Args:
            X (numpy.ndarray): matrice d'entrée de forme (nx, m)
                nx est le nombre de caractéristiques en entrée du neurone
                m est le nombre d'exemples

        Returns:
            tuple: (A1, A2) les sorties activées des couches
            cachée et de sortie
        """
        # Calcul de la couche cachée
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        # Calcul de la couche de sortie
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calcule le coût du modèle en utilisant la régression logistique

        Args:
            Y (numpy.ndarray): étiquettes correctes pour les
            données d'entrée (1, m)
            A (numpy.ndarray): sortie activée du neurone de forme (1, m)

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
        Évalue les prédictions du réseau de neurones

        Args:
            X (numpy.ndarray): matrice d'entrée de forme (nx, m)
                nx est le nombre de caractéristiques
                m est le nombre d'exemples
            Y (numpy.ndarray): étiquettes correctes de forme (1, m)

        Returns:
            tuple: (prédictions, coût)
                prédictions (numpy.ndarray): vecteur de forme (1, m)
                    contenant les prédictions pour chaque exemple
                coût (float): coût du modèle
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        predictions = np.where(self.__A2 < 0.5, 0, 1)
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calcule une passe de descente de gradient sur le réseau de neurones

        Args:
            X (numpy.ndarray): données d'entrée de forme (nx, m)
            Y (numpy.ndarray): étiquettes correctes de forme (1, m)
            A1 (numpy.ndarray): sortie de la couche cachée
            A2 (numpy.ndarray): sortie prédite
            alpha (float): taux d'apprentissage
        """
        m = X.shape[1]

        # Calcul des gradients de la couche de sortie
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # Calcul des gradients de la couche cachée
        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Mise à jour des poids et biais
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
        
    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, 
              graph=True, step=100):
        """
        Entraîne le réseau de neurones
        
        Args:
            X (numpy.ndarray): données d'entrée (nx, m)
            Y (numpy.ndarray): étiquettes correctes (1, m)
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
            # Propagation avant
            A1, A2 = self.forward_prop(X)
            
            # Affichage du coût
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A2)
                costs.append(cost)
                steps.append(i)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
            
            if i < iterations:
                # Descente de gradient
                self.gradient_descent(X, Y, A1, A2, alpha)

        # Affichage du graphique
        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
