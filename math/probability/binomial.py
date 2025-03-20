#!/usr/bin/env python3
"""
Module contenant la classe Binomial qui représente
une distribution de probabilité binomiale.
"""


class Binomial:
    """
    Classe représentant une distribution binomiale.
    
    Une distribution binomiale modélise le nombre de succès dans une séquence
    de n essais indépendants, chacun ayant la même probabilité p de succès.
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialise une instance de distribution binomiale.
        
        Args:
            data (list, optional): Liste de données pour estimer la distribution
            n (int, optional): Nombre d'essais de Bernoulli
            p (float, optional): Probabilité de succès pour chaque essai
        
        Raises:
            ValueError: Si n n'est pas une valeur positive
            ValueError: Si p n'est pas une probabilité valide
            TypeError: Si data n'est pas une liste
            ValueError: Si data ne contient pas au moins deux valeurs
        """
        if data is None:
            # Si aucune donnée n'est fournie, utiliser les n et p spécifiés
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            # Si des données sont fournies, calculer n et p à partir des données
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            
            # Calculer la moyenne et la variance des données
            mean = sum(data) / len(data)
            variance = sum([(x - mean) ** 2 for x in data]) / len(data)
            
            # Calculer p à partir de la moyenne et de la variance
            # Pour une distribution binomiale, on a: mean = n*p et variance = n*p*(1-p)
            # Donc p = 1 - (variance / mean)
            p = 1 - (variance / mean)
            
            # Calculer n à partir de la moyenne et de p
            # mean = n*p, donc n = mean/p
            n = mean / p
            
            # Arrondir n à l'entier le plus proche
            n = round(n)
            
            # Recalculer p à partir de n et de la moyenne pour assurer la cohérence
            p = mean / n
            
            # Sauvegarder n et p comme attributs d'instance
            self.n = int(n)
            self.p = float(p)
    
    def pmf(self, k):
        """
        Calcule la valeur de la fonction de masse de probabilité (PMF)
        pour un nombre donné de "succès".
        
        Args:
            k: Le nombre de "succès"
            
        Returns:
            La valeur PMF pour k
        """
        # Convertir k en entier s'il ne l'est pas déjà
        if not isinstance(k, int):
            k = int(k)
        
        # Si k est hors plage, retourner 0
        if k < 0 or k > self.n:
            return 0
        
        # Calculer le coefficient binomial C(n,k)
        # C(n,k) = n! / (k! × (n-k)!)
        # Pour éviter des calculs de factorielles très grands, on calcule directement
        # en multipliant/divisant de manière itérative
        
        coef = 1
        for i in range(1, k + 1):
            coef = coef * (self.n - (i - 1)) / i
        
        # Calculer p^k et (1-p)^(n-k)
        p_power_k = self.p ** k
        q_power_nk = (1 - self.p) ** (self.n - k)
        
        # Calculer la PMF: C(n,k) × p^k × (1-p)^(n-k)
        pmf_value = coef * p_power_k * q_power_nk
        
        return pmf_value
    
    def cdf(self, k):
        """
        Calcule la valeur de la fonction de distribution cumulative (CDF)
        pour un nombre donné de "succès".
        
        Args:
            k: Le nombre de "succès"
            
        Returns:
            La valeur CDF pour k
        """
        # Convertir k en entier s'il ne l'est pas déjà
        if not isinstance(k, int):
            k = int(k)
        
        # Si k est hors plage inférieure, retourner 0
        if k < 0:
            return 0
        
        # Si k est supérieur ou égal à n, retourner 1
        if k >= self.n:
            return 1.0
        
        # Calculer la CDF en utilisant la fonction pmf
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)
        
        return cdf_value