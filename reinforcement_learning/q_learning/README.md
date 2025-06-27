# Q-learning

![Holberton](https://www.holbertonschool.com/holberton-logo.png)

Projet réalisé dans le cadre de la formation à Holberton School.

## Description

Ce projet implémente un agent qui apprend à jouer au jeu FrozenLake en utilisant l'algorithme Q-learning. Le Q-learning est un algorithme d'apprentissage par renforcement qui permet à un agent d'apprendre la meilleure stratégie (politique) pour maximiser ses récompenses cumulées dans un environnement donné.

### Concepts clés

- **Environnement** : FrozenLake, un jeu où l'agent doit naviguer sur une surface gelée pour atteindre un objectif sans tomber dans des trous.
- **Agent** : L'entité qui prend des actions dans l'environnement.
- **Q-table** : Une table qui stocke les valeurs Q pour chaque paire état-action.
- **Epsilon-greedy** : Stratégie pour équilibrer exploration et exploitation.
- **Apprentissage par renforcement** : L'agent apprend en interagissant avec l'environnement pour maximiser une récompense cumulative.

## Fichiers et structure

Le projet est composé des fichiers suivants :

1. `0-load_env.py` : Charge l'environnement FrozenLake.
2. `1-q_init.py` : Initialise la Q-table.
3. `2-epsilon_greedy.py` : Implémente la stratégie epsilon-greedy pour choisir des actions.
4. `3-train.py` : Entraîne l'agent en utilisant le Q-learning.
5. `4-play.py` : Joue une partie en utilisant la Q-table entraînée.
6. `README.md` : Ce fichier, qui décrit le projet.

## Fonctionnalités

### 1. Chargement de l'environnement

Le fichier `0-load_env.py` contient la fonction `load_frozen_lake` qui charge l'environnement FrozenLake avec différentes configurations possibles :

- Taille de la grille personnalisée.
- Carte prédéfinie.
- Glissance de la surface.

### 2. Initialisation de la Q-table

Le fichier `1-q_init.py` contient la fonction `q_init` qui initialise la Q-table avec des zéros. La taille de la Q-table est déterminée par le nombre d'états et d'actions dans l'environnement.

### 3. Stratégie epsilon-greedy

Le fichier `2-epsilon_greedy.py` contient la fonction `epsilon_greedy` qui implémente la stratégie epsilon-greedy :

- Avec une probabilité epsilon, l'agent explore en choisissant une action aléatoire.
- Avec une probabilité 1-epsilon, l'agent exploite en choisissant l'action avec la plus haute valeur Q.

### 4. Entraînement de l'agent

Le fichier `3-train.py` contient la fonction `train` qui implémente l'algorithme Q-learning pour entraîner l'agent :

- Met à jour la Q-table en fonction des récompenses reçues et des transitions d'état.
- Utilise epsilon-greedy pour l'exploration/exploitation pendant l'entraînement.
- Décroît epsilon au fil du temps pour passer de l'exploration à l'exploitation.

### 5. Jeu avec la Q-table entraînée

Le fichier `4-play.py` contient la fonction `play` qui utilise la Q-table entraînée pour jouer une partie de FrozenLake :

- L'agent choisit toujours l'action avec la plus haute valeur Q (exploitation pure).
- Affiche le chemin suivi par l'agent pour atteindre l'objectif.

## Auteurs

Ce projet a été réalisé par Samuel VERSCHUEREN dans le cadre de la formation à Holberton School.

```

```
