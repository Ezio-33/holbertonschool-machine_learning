#!/usr/bin/env python3
"""
Module contenant la classe Exponential qui représente
une distribution de probabilité exponentielle.
"""


class Exponential:
    """
    Classe représentant une distribution exponentielle.
    
    Une distribution exponentielle modélise le temps d'attente
    entre des événements qui se produisent à un taux constant.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialise une instance de distribution exponentielle.
        
        Args:
            data (list, optional): Liste de données pour estimer la distribution
            lambtha (float, optional): Taux d'occurrence des événements
        
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
            # Si des données sont fournies, calculer lambtha à partir des données
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            
            # Pour une distribution exponentielle, lambtha est l'inverse de la moyenne
            mean = sum(data) / len(data)
            self.lambtha = float(1 / mean)

    def pdf(self, x):
        """
        Calcule la valeur de la fonction de densité de probabilité (PDF)
        pour une période de temps donnée.
        
        Args:
            x: La période de temps
            
        Returns:
            La valeur PDF pour x
        """
        # Constante e approximative
        e = 2.7182818285
        
        # Si x est négatif (hors plage), retourner 0
        if x < 0:
            return 0
        
        # Calculer la PDF pour la distribution exponentielle: λ * e^(-λx)
        pdf_value = self.lambtha * (e ** (-self.lambtha * x))
        
        return pdf_value

    def cdf(self, x):
        """
        Calcule la valeur de la fonction de distribution cumulative (CDF)
        pour une période de temps donnée.
        
        Args:
            x: La période de temps
            
        Returns:
            La valeur CDF pour x
        """
        # Constante e approximative
        e = 2.7182818285
        
        # Si x est négatif (hors plage), retourner 0
        if x < 0:
            return 0
        
        # Calculer la CDF pour la distribution exponentielle: 1 - e^(-λx)
        cdf_value = 1 - (e ** (-self.lambtha * x))
        
        return cdf_value