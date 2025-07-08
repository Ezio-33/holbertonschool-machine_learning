# Apprentissage par Différences Temporelles

Ce projet implémente divers algorithmes d'apprentissage par différences temporelles pour l'apprentissage par renforcement, en se concentrant spécifiquement sur la méthode Monte Carlo pour l'estimation des fonctions de valeur.

## Description

L'apprentissage par différences temporelles (TD) est un concept central en apprentissage par renforcement qui combine les idées des méthodes Monte Carlo et de la programmation dynamique. Ce projet explore ces concepts en utilisant l'environnement FrozenLake de Gymnasium.

## Environnement

- **Plateforme**: Ubuntu 20.04 LTS
- **Version Python**: 3.9
- **Dépendances**:
  - numpy (version 1.25.2)
  - gymnasium (version 0.29.1)

## Fichiers

### 0-monte_carlo.py

Implémente l'algorithme Monte Carlo pour l'estimation des fonctions de valeur.

**Fonctions:**

- `sample_episode(env, policy, max_steps=100)`: Simule un épisode complet en suivant une politique donnée
- `monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)`: Exécute l'algorithme Monte Carlo pour l'estimation de la fonction de valeur

**Paramètres:**

- `env`: Instance de l'environnement (FrozenLake8x8-v1)
- `V`: Tableau numpy contenant les estimations de valeur
- `policy`: Fonction qui prend un état et retourne la prochaine action
- `episodes`: Nombre total d'épisodes pour l'entraînement
- `max_steps`: Nombre maximum d'étapes par épisode
- `alpha`: Taux d'apprentissage
- `gamma`: Taux d'actualisation

## Utilisation

```bash
python3 0-main.py
```

## Objectifs d'Apprentissage

Après avoir terminé ce projet, vous devriez être capable d'expliquer :

- Qu'est-ce que la méthode Monte Carlo en apprentissage par renforcement
- Qu'est-ce que l'apprentissage par différences temporelles
- Qu'est-ce que le bootstrapping
- Qu'est-ce que les différences temporelles n-step
- Qu'est-ce que TD(λ)
- Qu'est-ce qu'une trace d'éligibilité
- Qu'est-ce que SARSA, SARSA(λ), et SARSAMAX
- La différence entre les méthodes 'on-policy' et 'off-policy'

## Sortie Attendue

L'algorithme Monte Carlo devrait converger vers des valeurs similaires à :

```
[[ 0.4305  0.729   0.6561  0.729   0.729   0.9     0.5905  0.5314]
 [ 0.2542  0.5314  0.5905  0.81    0.6561  0.3874  0.4783  0.3874]
 [ 0.729   0.2824  0.3487 -1.      1.      0.4783  0.4305  0.4305]
 [ 1.      0.4305  0.2288  0.5905  0.9    -1.      0.4783  0.4783]
 [ 1.      0.6561  0.5905 -1.      1.      1.      0.729   0.729 ]
 [ 1.     -1.     -1.      1.      1.      1.     -1.      0.9   ]
 [ 1.     -1.      1.      1.     -1.      1.     -1.      1.    ]
 [ 1.      1.      1.     -1.      1.      1.      1.      1.    ]]
```

## Auteur

Ce projet fait partie du curriculum Machine Learning de l'École Holberton.

## Dépôt

- **Dépôt GitHub**: holbertonschool-machine_learning
- **Répertoire**: reinforcement_learning/temporal_difference
