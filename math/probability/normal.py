#!/usr/bin/env python3
"""
Module contenant la classe Normal qui représente
une distribution de probabilité normale.
"""


class Normal:
    """
    Classe représentant une distribution normale.

    Une distribution normale modélise des phénomènes dont les valeurs
    se répartissent symétriquement autour d'une valeur centrale (la moyenne),
    avec une dispersion caractérisée par l'écart-type.
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialise une instance de distribution normale.

        Args:
            data (list, optional): Liste de données pour estimer
            la distribution mean (float, optional): Moyenne de la distribution
            stddev (float, optional): Écart-type de la distribution

        Raises:
            ValueError: Si stddev n'est pas une valeur positive
            TypeError: Si data n'est pas une liste
            ValueError: Si data ne contient pas au moins deux valeurs
        """
        if data is None:
            # Si aucune donnée n'est fournie, utiliser
            # les mean et stddev spécifiés
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            # Si des données sont fournies, calculer mean
            # et stddev à partir des données
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculer la moyenne
            self.mean = float(sum(data) / len(data))

            # Calculer l'écart-type
            variance = sum([(x - self.mean) ** 2 for x in data]) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """
        Calcule le z-score d'une valeur x donnée.

        Args:
            x: La valeur x dont on veut calculer le z-score

        Returns:
            Le z-score de x
        """
        # Le z-score est calculé comme (x - μ) / σ
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calcule la valeur x correspondant à un z-score donné.

        Args:
            z: Le z-score dont on veut calculer la valeur x correspondante

        Returns:
            La valeur x correspondant au z-score z
        """
        # La valeur x est calculée comme μ + z * σ
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calcule la valeur de la fonction de densité de probabilité (PDF)
        pour une valeur x donnée.

        Args:
            x: La valeur x dont on veut calculer la PDF

        Returns:
            La valeur PDF pour x
        """
        # Constantes π et e
        pi = 3.1415926536
        e = 2.7182818285

        # Calculer la PDF pour la distribution normale
        coefficient = 1 / (self.stddev * ((2 * pi) ** 0.5))
        exponent = -((x - self.mean) ** 2) / (2 * (self.stddev ** 2))

        return coefficient * (e ** exponent)

    def cdf(self, x):
        """
        Calcule la valeur de la fonction de distribution cumulative (CDF)
        pour une valeur x donnée.

        Args:
            x: La valeur x dont on veut calculer la CDF

        Returns:
            La valeur CDF pour x
        """
        # Utiliser la relation entre CDF et erf
        return 0.5 * (1 + self.erf(
            (x - self.mean) / (self.stddev * (2 ** 0.5))
        ))

    def erf(self, x):
        """
        Approximation de la fonction d'erreur (erf) par une série de Taylor.

        Args:
            x: La valeur pour laquelle calculer erf

        Returns:
            L'approximation de erf(x)
        """
        pi = 3.1415926536

        # Approximation par série de Taylor (plus précise pour ce projet)
        return (2 / (pi ** 0.5)) * (
            x - (x**3 / 3) + (x**5 / 10) - (x**7 / 42) + (x**9 / 216)
        )
