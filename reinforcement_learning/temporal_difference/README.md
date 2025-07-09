# Apprentissage par Différences Temporelles

Ce projet implémente divers algorithmes d'apprentissage par différences temporelles pour l'apprentissage par renforcement, incluant Monte Carlo, TD(λ) et SARSA(λ) pour l'estimation des fonctions de valeur et l'apprentissage de politiques.

## Description

L'apprentissage par différences temporelles (TD) est un concept central en apprentissage par renforcement qui combine les idées des méthodes Monte Carlo et de la programmation dynamique. Ce projet explore ces concepts en utilisant l'environnement FrozenLake de Gymnasium, un environnement de grille où l'agent doit naviguer sur un lac gelé pour atteindre un objectif tout en évitant les trous.

## Environnement

- **Plateforme**: Ubuntu 20.04 LTS
- **Version Python**: 3.9
- **Dépendances**:
  - numpy (version 1.25.2)
  - gymnasium (version 0.29.1)

## Fichiers

### 0-monte_carlo.py

Implémente l'algorithme Monte Carlo pour l'estimation des fonctions de valeur d'état.

**Fonctions :**

- `sample_episode(env, policy, max_steps=100)`: Simule un épisode complet en suivant une politique donnée
- `monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)`: Exécute l'algorithme Monte Carlo pour l'estimation de la fonction de valeur

**Paramètres :**

- `env`: Instance de l'environnement (FrozenLake8x8-v1)
- `V`: Tableau numpy contenant les estimations de valeur d'état
- `policy`: Fonction qui prend un état et retourne la prochaine action
- `episodes`: Nombre total d'épisodes pour l'entraînement (défaut: 5000)
- `max_steps`: Nombre maximum d'étapes par épisode (défaut: 100)
- `alpha`: Taux d'apprentissage (défaut: 0.1)
- `gamma`: Taux d'actualisation (défaut: 0.99)

### 1-td_lambtha.py

Implémente l'algorithme TD(λ) avec traces d'éligibilité pour l'estimation des fonctions de valeur d'état.

**Fonctions :**

- `td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)`: Exécute l'algorithme TD(λ) pour l'estimation de la fonction de valeur

**Paramètres :**

- `env`: Instance de l'environnement (FrozenLake8x8-v1)
- `V`: Tableau numpy contenant les estimations de valeur d'état
- `policy`: Fonction qui prend un état et retourne la prochaine action
- `lambtha`: Facteur de trace d'éligibilité λ (entre 0 et 1)
- `episodes`: Nombre total d'épisodes pour l'entraînement (défaut: 5000)
- `max_steps`: Nombre maximum d'étapes par épisode (défaut: 100)
- `alpha`: Taux d'apprentissage (défaut: 0.1)
- `gamma`: Taux d'actualisation (défaut: 0.99)

**Retour :**

- `V`: Tableau numpy mis à jour contenant les nouvelles estimations de valeur d'état

### 2-sarsa_lambtha.py

Implémente l'algorithme SARSA(λ) pour l'apprentissage par renforcement avec politique epsilon-greedy et traces d'éligibilité.

**Fonctions :**

- `sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)`: Exécute l'algorithme SARSA(λ) pour l'apprentissage de la table Q
- `get_action(state, Q, epsilon)`: Fonction interne implémentant la politique epsilon-greedy

**Paramètres :**

- `env`: Instance de l'environnement (FrozenLake8x8-v1)
- `Q`: Tableau numpy de forme (s,a) contenant la table Q (valeurs action-état)
- `lambtha`: Facteur de trace d'éligibilité λ (entre 0 et 1)
- `episodes`: Nombre total d'épisodes pour l'entraînement (défaut: 5000)
- `max_steps`: Nombre maximum d'étapes par épisode (défaut: 100)
- `alpha`: Taux d'apprentissage (défaut: 0.1)
- `gamma`: Taux d'actualisation (défaut: 0.99)
- `epsilon`: Seuil initial pour epsilon greedy (défaut: 1.0)
- `min_epsilon`: Valeur minimale vers laquelle epsilon doit décroître (défaut: 0.1)
- `epsilon_decay`: Taux de décroissance exponentielle pour epsilon (défaut: 0.05)

**Caractéristiques de l'implémentation :**

- Utilise une décroissance exponentielle d'epsilon : `epsilon = min_epsilon + (initial_epsilon - min_epsilon) * exp(-epsilon_decay * episode)`
- Réinitialise les traces d'éligibilité à chaque épisode
- Met à jour toutes les valeurs Q simultanément via les traces d'éligibilité
- Gère correctement les états terminaux

**Retour :**

- `Q`: Tableau numpy mis à jour contenant la table Q optimisée

## Utilisation

Pour exécuter Monte Carlo :

````bash
python3 0-main.py
## Utilisation

Pour exécuter Monte Carlo :

```bash
python3 0-main.py
````

Pour exécuter TD(λ) :

```bash
python3 1-main.py
```

Pour exécuter SARSA(λ) :

```bash
python3 2-main.py
```

## Concepts Théoriques

### Monte Carlo

- **Méthode d'estimation** basée sur l'échantillonnage d'épisodes complets
- **Attente jusqu'à la fin** de l'épisode pour mettre à jour les valeurs
- **Variance élevée** mais **sans biais**
- Utilise les retours réels (somme actualisée des récompenses)

### TD(λ) (Temporal Difference Lambda)

- **Combinaison** entre Monte Carlo et programmation dynamique
- **Traces d'éligibilité** permettent de propager les mises à jour
- **Bootstrapping** : utilise les estimations actuelles pour mettre à jour
- **Paramètre λ** contrôle le compromis entre TD(0) et Monte Carlo

### SARSA(λ) (State-Action-Reward-State-Action Lambda)

- **Méthode on-policy** : apprend la politique qu'elle suit
- **Table Q** : apprend les valeurs action-état au lieu des valeurs d'état
- **Politique epsilon-greedy** avec décroissance exponentielle
- **Traces d'éligibilité** pour accélérer l'apprentissage

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

### Monte Carlo

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

### TD(λ)

L'algorithme TD(λ) devrait converger vers des valeurs similaires à :

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

### SARSA(λ)

L'algorithme SARSA(λ) produit une table Q de forme (64, 4) représentant les valeurs action-état. Exemple de sortie avec seed=0 et λ=0.9 :

```
[[0.6726 0.7283 0.6846 0.6825]
 [0.6311 0.7332 0.6026 0.6248]
 [0.6692 0.7022 0.6466 0.5803]
 [0.6244 0.6988 0.6238 0.6073]
 [0.6673 0.694  0.7188 0.6805]
 [0.674  0.7216 0.7199 0.7059]
 [0.7067 0.7287 0.6775 0.7028]
 [0.7151 0.6909 0.7225 0.7427]
 ...]
```

**Note** : Les valeurs peuvent varier légèrement selon l'implémentation et les graines aléatoires, mais devraient converger vers des valeurs dans des plages similaires.

## Comparaison des Algorithmes

| Algorithme      | Type      | Mise à jour   | Variance | Biais  | Vitesse de convergence |
| --------------- | --------- | ------------- | -------- | ------ | ---------------------- |
| **Monte Carlo** | Off-line  | Fin d'épisode | Élevée   | Aucun  | Lente                  |
| **TD(λ)**       | On-line   | Chaque étape  | Modérée  | Faible | Rapide                 |
| **SARSA(λ)**    | On-policy | Chaque étape  | Modérée  | Faible | Rapide                 |

## Notes d'Implémentation

### Paramètres recommandés

- **alpha (taux d'apprentissage)** : 0.1 - 0.3
- **gamma (facteur d'actualisation)** : 0.95 - 0.99
- **lambda (traces d'éligibilité)** : 0.8 - 0.95
- **epsilon initial** : 1.0 (exploration complète au début)
- **epsilon final** : 0.1 (10% d'exploration à la fin)

### Conseils de débogage

1. Vérifiez que les traces d'éligibilité sont correctement réinitialisées
2. Assurez-vous que epsilon décroît progressivement
3. Validez que les états terminaux sont gérés correctement
4. Utilisez des graines fixes pour la reproductibilité
   [0.9755 0.8558 0.0117 0.36 ]
   [0.73 0.1716 0.521 0.0543]
   [0.2 0.1573 0.7536 0.3724]
   [0.4364 0.8292 0.7044 0.2497]
   [0.4121 0.8949 0.612 0.2379]
   [0.9342 0.614 0.5356 0.5899]
   [1.0864 0.631 0.4417 0.5178]
   [0.2747 0.4748 0.4896 0.504 ]
   [0.2274 0.2544 0.058 0.4344]
   [0.3118 0.6575 0.3778 0.1796]
   [0.0247 0.0672 0.6583 0.4537]
   [0.5366 0.8967 0.9903 0.2169]
   [0.7139 0.2633 0.0207 0.8476]
   [0.32 0.3835 0.5883 0.831 ]
   [0.7608 1.3898 0.6772 0.798 ]
   [0.3321 0.5003 0.423 0.2845]
   [0.5159 0.4794 0.3253 0.2545]
   [0.4841 0.0257 0.2649 0.4273]
   [0.3742 0.4636 0.2776 0.5868]
   [0.8639 0.1175 0.5174 0.1321]
   [0.7169 0.3961 0.5654 0.1833]
   [0.1448 0.4881 0.3556 0.9404]
   [0.7653 0.7487 0.9037 0.0834]]

```

## Structure du Projet

```

temporal_difference/
├── README.md # Documentation du projet
├── 0-monte_carlo.py # Implémentation Monte Carlo
├── 1-td_lambtha.py # Implémentation TD(λ)
├── 2-sarsa_lambtha.py # Implémentation SARSA(λ)
├── 0-main.py # Script de test Monte Carlo
├── 1-main.py # Script de test TD(λ)
└── 2-main.py # Script de test SARSA(λ)

````

## Prérequis

Avant d'exécuter les scripts, assurez-vous d'avoir :

1. **Python 3.9** ou version supérieure
2. **numpy** : `pip install numpy`
3. **gymnasium** : `pip install gymnasium`
4. **Environnement activé** : `pyenv activate formation-env` (si applicable)

## Tests et Validation

Pour valider vos implémentations :

```bash
# Vérifier le style de code
pycodestyle *.py

# Rendre les fichiers exécutables
chmod +x *.py

# Exécuter les tests
./0-main.py  # Monte Carlo
./1-main.py  # TD(λ)
./2-main.py  # SARSA(λ)
````

## Auteur

Ce projet fait partie du curriculum Machine Learning de l'École Holberton.

## Dépôt

- **Dépôt GitHub**: holbertonschool-machine_learning
- **Répertoire**: reinforcement_learning/temporal_difference
