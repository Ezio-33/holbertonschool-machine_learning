#!/usr/bin/env python3
"""fonction qui calcule le coût avec régularisation L2"""

import numpy as np

def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calcule le coût d'un réseau de neurones avec régularisation L2
    
    Parameters:
        cost (float): coût initial sans régularisation
        lambtha (float): paramètre de régularisation
        weights (dict): dictionnaire des poids du réseau
        L (int): nombre de couches
        m (int): nombre d'exemples d'entraînement
        
    Returns:
        float: coût total avec régularisation L2
    """
    # Initialisation du terme de régularisation
    l2_reg = 0
    
    # Parcours de chaque couche pour sommer les carrés des poids
    for i in range(1, L + 1):
        l2_reg += np.sum(np.square(weights['W' + str(i)]))
        
    # Calcul du coût final avec régularisation
    l2_reg_cost = cost + (lambtha / (2 * m)) * l2_reg
    
    return l2_reg_cost
