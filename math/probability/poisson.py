#!/usr/bin/env python3
"""
Module contenant la classe Poisson qui représente
une distribution de probabilité de Poisson.
"""


class Poisson:
    """
    Classe représentant une distribution de Poisson.

    Une distribution de Poisson modélise le nombre d'événements
    qui se produisent dans un intervalle fixe avec un taux moyen connu.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialise une instance de distribution de Poisson.

        Args:
            data (list, optional): Liste de données pour estimer
            la distribution lambtha (float, optional): Nombre moyen
            d'occurrences dans un intervalle

        Raises:
            ValueError: Si lambtha n'est pas une valeur positive
            TypeError: Si data n'est pas une liste
            ValueError: Si data ne contient pas au moins deux valeurs
        """
        if data is None:
            # Si aucune donnée n'est fournie, utiliser le lambtha spécifié
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Si des données sont fournies, calculer
            # lambtha à partir des données
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculer lambtha comme la moyenne des données
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calcule la valeur de la fonction de masse de probabilité (PMF)
        pour un nombre donné de "succès".

        Args:
            k: Le nombre de "succès" (événements)

        Returns:
            La valeur PMF pour k
        """
        # Constante e approximative
        e = 2.7182818285

        # Convertir k en entier s'il ne l'est pas déjà
        if not isinstance(k, int):
            k = int(k)

        # Si k est négatif, retourner 0
        if k < 0:
            return 0

        # Calculer la factorielle de k
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # Calculer la PMF de Poisson: (e^(-λ) * λ^k) / k!
        pmf_value = (e ** (-self.lambtha) * (self.lambtha ** k)) / factorial

        return pmf_value

    def cdf(self, k):
        """
        Calcule la valeur de la fonction de distribution cumulative (CDF)
        pour un nombre donné de "succès".

        Args:
            k: Le nombre de "succès" (événements)

        Returns:
            La valeur CDF pour k
        """
        # Convertir k en entier s'il ne l'est pas déjà
        if not isinstance(k, int):
            k = int(k)

        # Si k est négatif, retourner 0
        if k < 0:
            return 0

        # Calculer la CDF comme la somme des PMF de 0 à k
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
