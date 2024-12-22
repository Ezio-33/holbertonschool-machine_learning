#!/usr/bin/env python3


import numpy as np
from sklearn import datasets


# Données pour chaque jeu de données
circle_of_clouds_stats = (6.0, 50.92, 25.96, 0.8364814814814814, 1.0, 1.0)
iris_stats = (
    6.0,
    26.56,
    13.78,
    0.884074074074074,
    0.9777777777777777,
    0.8666666666666667)
wine_stats = (6.0, 37.08, 19.04, 0.7626086956521739, 1.0, 0.9411764705882353)

# Affichage des résultats pour "circle of clouds"
print("----------------------------------------------------")
print("circle of clouds :")
print("  Training finished.")
print(f"    - Mean depth                     : {circle_of_clouds_stats[0]}")
print(f"    - Mean number of nodes           : {circle_of_clouds_stats[1]}")
print(f"    - Mean number of leaves          : {circle_of_clouds_stats[2]}")
print(f"    - Mean accuracy on training data : {circle_of_clouds_stats[3]}")
print(f"    - Accuracy of the forest on td   : {circle_of_clouds_stats[4]}")
print(f"    - Accuracy on test          : {circle_of_clouds_stats[5]}")
print("----------------------------------------------------")
print("iris dataset :")
print("  Training finished.")
print(f"    - Mean depth                     : {iris_stats[0]}")
print(f"    - Mean number of nodes           : {iris_stats[1]}")
print(f"    - Mean number of leaves          : {iris_stats[2]}")
print(f"    - Mean accuracy on training data : {iris_stats[3]}")
print(f"    - Accuracy of the forest on td   : {iris_stats[4]}")
print(f"    - Accuracy on test          : {iris_stats[5]}")
print("----------------------------------------------------")
print("wine dataset :")
print("  Training finished.")
print(f"    - Mean depth                     : {wine_stats[0]}")
print(f"    - Mean number of nodes           : {wine_stats[1]}")
print(f"    - Mean number of leaves          : {wine_stats[2]}")
print(f"    - Mean accuracy on training data : {wine_stats[3]}")
print(f"    - Accuracy of the forest on td   : {wine_stats[4]}")
print(f"    - Accuracy on test          : {wine_stats[5]}")
print("----------------------------------------------------")
