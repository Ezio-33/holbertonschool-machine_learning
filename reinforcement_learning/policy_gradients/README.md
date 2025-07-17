# Policy Gradients - Apprentissage par Renforcement

Ce projet implémente l'apprentissage par renforcement avec **Gradient de Politique** en utilisant l'algorithme du gradient de politique Monte-Carlo (**REINFORCE**).

## 📋 Description

Dans ce projet, nous implémentons notre propre Gradient de Politique dans la boucle d'apprentissage par renforcement en utilisant l'algorithme du gradient de politique Monte-Carlo - également appelé REINFORCE. Cette méthode permet à un agent d'apprendre une politique optimale en maximisant la récompense attendue grâce à l'optimisation basée sur le gradient.

### Concepts clés

- **Policy Gradient** : Méthode d'optimisation directe de la politique
- **REINFORCE** : Algorithme Monte-Carlo pour l'estimation du gradient de politique
- **Baseline** : Technique pour réduire la variance des estimations

## 🎯 Objectifs d'apprentissage

À la fin de ce projet, vous devriez être capable d'expliquer :

- **Qu'est-ce qu'une Politique ?** Une fonction qui mappe les états aux actions
- **Comment calculer un Gradient de Politique ?** Utilisation du théorème du gradient de politique
- **Qu'est-ce que le gradient de politique Monte-Carlo ?** Estimation par échantillonnage Monte-Carlo
- **Comment implémenter REINFORCE ?** Algorithme complet avec baseline optionnelle

## 🛠️ Prérequis

### Environnement système

- **OS** : Ubuntu 20.04 LTS
- **Python** : Version 3.9
- **Style** : Respect du standard `pycodestyle` (version 2.11.1)

### Dépendances Python

```bash
numpy==1.25.2
gymnasium==0.29.1
```

### Standards de code

- Tous les modules, classes et fonctions doivent être documentés
- Code conforme aux standards PEP 8
- Tests unitaires inclus

## 📁 Structure du projet

```
policy_gradients/
├── README.md                 # Documentation du projet
├── policy_gradient.py        # Implémentation du gradient de politique
├── train.py                  # Fonction d'entraînement REINFORCE
├── 0-main.py                 # Test de la fonction policy
├── 1-main.py                 # Test du gradient de politique Monte-Carlo
├── 2-main.py                 # Test de la fonction d'entraînement
└── 3-main.py                 # Test avec animation de l'entraînement
```

### Description des fichiers

| Fichier              | Description                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| `policy_gradient.py` | Contient les fonctions `policy` et `policy_gradient` pour l'implémentation du gradient de politique |
| `train.py`           | Contient la fonction d'entraînement complète pour l'algorithme REINFORCE                            |
| `0-main.py`          | Script de test pour la fonction policy simple                                                       |
| `1-main.py`          | Script de test pour le gradient de politique Monte-Carlo                                            |
| `2-main.py`          | Script de test pour la fonction d'entraînement                                                      |
| `3-main.py`          | Script de test avec visualisation de l'entraînement                                                 |

## 🚀 Installation et utilisation

### Installation des dépendances

```bash
pip install python==3.9 numpy==1.25.2 gymnasium==0.29.1
```

### Exécution des tests

```bash
# Tester la fonction policy simple
./0-main.py

# Tester le gradient de politique Monte-Carlo
./1-main.py

# Tester la fonction d'entraînement
./2-main.py

# Tester l'entraînement avec animation
./3-main.py
```

### Utilisation des modules

```python
from policy_gradient import policy, policy_gradient
from train import train

# Exemple d'utilisation
# Voir les fichiers main pour des exemples détaillés
```

## 📊 Algorithme REINFORCE

L'algorithme REINFORCE suit ces étapes principales :

1. **Initialisation** : Paramètres de politique θ
2. **Collecte de données** : Génération d'épisodes selon π_θ
3. **Calcul des retours** : Estimation Monte-Carlo des récompenses
4. **Mise à jour** : Gradient ascent sur log π_θ(a|s) \* G_t
5. **Répétition** : Jusqu'à convergence

## 🔍 Résultats attendus

- Convergence de la politique vers l'optimal
- Amélioration progressive des récompenses
- Stabilité de l'apprentissage avec baseline

## 👨‍💻 Auteur

**Samuel VERSCHUEREN**

## 📄 Licence

Ce projet fait partie du curriculum Holberton School.
