# GitHub Copilot: # Modèles de Markov Cachés (HMM)

<div align="left">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
</div>

---

## Table des Matières

- Aperçu
- Fonctionnalités
- Structure du Projet
- Prérequis
- Utilisation

---

## Aperçu

Ce projet implémente les algorithmes fondamentaux des Modèles de Markov Cachés (HMM) en Python. Les HMM sont des modèles statistiques où un système est modélisé comme un processus markovien avec des états cachés. Ce projet fournit des implémentations pour l'analyse des chaînes de Markov et des algorithmes essentiels pour les HMM.

---

## Fonctionnalités

- **Chaînes de Markov classiques**

  - Calcul des distributions de probabilités d'états après t transitions
  - Détermination de la régularité des chaînes de Markov
  - Identification des chaînes absorbantes

- **Modèles de Markov Cachés**
  - Algorithme Forward pour calculer la vraisemblance des observations
  - Algorithme de Viterbi pour déterminer la séquence d'états cachés la plus probable
  - Algorithme Backward pour le calcul des probabilités arrière
  - Algorithme de Baum-Welch pour l'apprentissage des paramètres du modèle

---

## Structure du Projet

```sh
└── /
    ├── 0-markov_chain.py    # Calcule les probabilités des états après t transitions
    ├── 1-regular.py         # Détermine les probabilités stationnaires d'une chaîne régulière
    ├── 2-absorbing.py       # Vérifie si une chaîne de Markov est absorbante
    ├── 3-forward.py         # Implémente l'algorithme Forward pour les HMM
    ├── 4-viterbi.py         # Implémente l'algorithme de Viterbi pour les HMM
    ├── 5-backward.py        # Implémente l'algorithme Backward pour les HMM
    ├── 6-baum_welch.py      # Estime les paramètres d'un HMM avec l'algorithme de Baum-Welch
    └── *-main.py            # Fichiers de test pour chaque implémentation
```

---

## Prérequis

Ce projet nécessite les dépendances suivantes:

- **Python 3.9**
- **NumPy 1.25.2**

## Utilisation

Chaque fichier Python peut être exécuté individuellement pour tester la fonctionnalité correspondante:

```sh
# Exemple pour tester l'algorithme de Viterbi
python3 4-main.py
```

Les fichiers d'implémentation contiennent les fonctions suivantes:

- **0-markov_chain.py**: `markov_chain(P, s, t=1)` - Calcule la distribution de probabilité après t étapes
- **1-regular.py**: `regular(P)` - Détermine les probabilités stationnaires d'une chaîne régulière
- **2-absorbing.py**: `absorbing(P)` - Détermine si une chaîne de Markov est absorbante
- **3-forward.py**: `forward(Observation, Emission, Transition, Initial)` - Implémente l'algorithme Forward
- **4-viterbi.py**: `viterbi(Observation, Emission, Transition, Initial)` - Trouve la séquence d'états cachés la plus probable
- **5-backward.py**: `backward(Observation, Emission, Transition, Initial)` - Implémente l'algorithme Backward
- **6-baum_welch.py**: `baum_welch(Observations, Transition, Emission, Initial, iterations=1000)` - Estime les paramètres d'un HMM

---
