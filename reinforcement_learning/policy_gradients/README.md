# Policy Gradients

Ce projet implémente l'apprentissage par renforcement avec Gradient de Politique en utilisant l'algorithme du gradient de politique Monte-Carlo (REINFORCE).

## Description

Dans ce projet, nous implémentons notre propre Gradient de Politique dans la boucle d'apprentissage par renforcement en utilisant l'algorithme du gradient de politique Monte-Carlo - également appelé REINFORCE.

## Objectifs d'apprentissage

- Qu'est-ce qu'une Politique?
- Comment calculer un Gradient de Politique?
- Qu'est-ce que le gradient de politique Monte-Carlo et comment l'utiliser?

## Prérequis

- Tous les fichiers seront interprétés/compilés sur Ubuntu 20.04 LTS avec python3 (version 3.9)
- Les fichiers seront exécutés avec numpy (version 1.25.2) et gymnasium (version 0.29.1)
- Le code doit respecter le style pycodestyle (version 2.11.1)
- Tous les modules, classes et fonctions doivent être documentés

## Fichiers

- `policy_gradient.py`: Contient la fonction policy et la fonction policy_gradient
- `train.py`: Contient la fonction d'entraînement pour l'algorithme du gradient de politique

## Utilisation

Exécutez les fichiers principaux pour tester chaque implémentation:

```bash
./0-main.py  # Tester la fonction policy simple
./1-main.py  # Tester le gradient de politique Monte-Carlo
./2-main.py  # Tester la fonction d'entraînement
./3-main.py  # Tester l'entraînement avec animation
```
