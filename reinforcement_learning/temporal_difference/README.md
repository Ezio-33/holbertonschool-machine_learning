# Apprentissage par Différences Temporelles

Ce projet implémente divers algorithmes d'apprentissage par différences temporelles pour l'apprentissage par renforcement, en se concentrant spécifiquement sur la méthode Monte Carlo et TD(λ) pour l'estimation des fonctions de valeur.

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

**Fonctions :**

- `sample_episode(env, policy, max_steps=100)`: Simule un épisode complet en suivant une politique donnée
- `monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)`: Exécute l'algorithme Monte Carlo pour l'estimation de la fonction de valeur

**Paramètres :**

- `env`: Instance de l'environnement (FrozenLake8x8-v1)
- `V`: Tableau numpy contenant les estimations de valeur
- `policy`: Fonction qui prend un état et retourne la prochaine action
- `episodes`: Nombre total d'épisodes pour l'entraînement
- `max_steps`: Nombre maximum d'étapes par épisode
- `alpha`: Taux d'apprentissage
- `gamma`: Taux d'actualisation

### 1-td_lambtha.py

Implémente l'algorithme TD(λ) avec traces d'éligibilité pour l'estimation des fonctions de valeur.

**Fonctions :**

- `td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)`: Exécute l'algorithme TD(λ) pour l'estimation de la fonction de valeur.

**Paramètres :**

- `env`: Instance de l'environnement (FrozenLake8x8-v1)
- `V`: Tableau numpy contenant les estimations de valeur
- `policy`: Fonction qui prend un état et retourne la prochaine action
- `lambtha`: Facteur de trace d'éligibilité λ (entre 0 et 1)
- `episodes`: Nombre total d'épisodes pour l'entraînement
- `max_steps`: Nombre maximum d'étapes par épisode
- `alpha`: Taux d'apprentissage
- `gamma`: Taux d'actualisation

**Retour :**

- `V`: Tableau numpy mis à jour contenant les nouvelles estimations de valeur

### 2-sarsa_lambtha.py

Implémente l'algorithme SARSA(λ) pour l'apprentissage par renforcement avec politique epsilon-greedy.

**Fonctions :**

- `sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)`: Exécute l'algorithme SARSA(λ) pour l'apprentissage de la table Q.

**Paramètres :**

- `env`: Instance de l'environnement (FrozenLake8x8-v1)
- `Q`: Tableau numpy de forme (s,a) contenant la table Q
- `lambtha`: Facteur de trace d'éligibilité λ (entre 0 et 1)
- `episodes`: Nombre total d'épisodes pour l'entraînement
- `max_steps`: Nombre maximum d'étapes par épisode
- `alpha`: Taux d'apprentissage
- `gamma`: Taux d'actualisation
- `epsilon`: Seuil initial pour epsilon greedy
- `min_epsilon`: Valeur minimale vers laquelle epsilon doit décroître
- `epsilon_decay`: Taux de décroissance pour mettre à jour epsilon

**Retour :**

- `Q`: Tableau numpy mis à jour contenant la table Q

## Utilisation

Pour exécuter Monte Carlo :

```bash
python3 0-main.py
```

Pour exécuter TD(λ) :

```bash
python3 1-main.py
```

Pour exécuter SARSA(λ) :

```bash
python3 2-main.py
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

L'algorithme SARSA(λ) devrait converger vers des valeurs similaires à :

```
[[0.5917 0.5997 0.6283 0.5918]
 [0.5893 0.6355 0.5939 0.5845]
 [0.5568 0.572  0.688  0.5585]
 [0.5545 0.5946 0.7661 0.6201]
 [0.6226 0.6195 0.7369 0.614 ]
 [0.6931 0.6793 0.8213 0.6718]
 [0.7971 0.6918 0.6703 0.6686]
 [0.8024 0.6791 0.6563 0.6219]
 [0.6125 0.6351 0.6125 0.6422]
 [0.5864 0.6039 0.5875 0.5874]
 [0.5664 0.6017 0.5479 0.6747]
 [0.5146 0.4951 0.5153 0.7284]
 [0.5505 0.634  0.5228 0.7369]
 [0.6297 0.6751 0.8142 0.663 ]
 [0.6484 0.8341 0.6937 0.5979]
 [0.6467 0.6218 0.8103 0.6655]
 [0.5993 0.7015 0.6193 0.6334]
 [0.5971 0.6108 0.5868 0.5976]
 [0.6443 0.548  0.4763 0.4793]
 [0.2828 0.1202 0.2961 0.1187]
 [0.6174 0.4852 0.5329 0.4691]
 [0.729  0.6998 0.8133 0.5891]
 [0.7403 0.8437 0.7401 0.7181]
 [0.713  0.7918 0.7071 0.7502]
 [0.639  0.6339 0.7207 0.665 ]
 [0.7195 0.6674 0.6615 0.6349]
 [0.6745 0.6791 0.6495 0.6417]
 [0.556  0.7595 0.6991 0.6854]
 [0.6474 0.6902 0.7873 0.647 ]
 [0.8811 0.5813 0.8817 0.6925]
 [0.8766 0.8101 0.7669 0.7591]
 [0.9115 0.708  0.6234 0.6611]
 [0.666  0.7145 0.6484 0.6715]
 [0.8512 0.6785 0.6583 0.547 ]
 [0.6784 0.6606 0.6792 0.6716]
 [0.8965 0.3676 0.4359 0.8919]
 [0.7818 0.8464 0.4823 0.7724]
 [0.7129 0.8106 0.3372 0.8471]
 [0.6463 0.7479 0.712  0.8417]
 [1.0457 0.7481 0.6679 0.447 ]
 [0.6663 0.6085 0.6239 0.7007]
 [0.9755 0.8558 0.0117 0.36  ]
 [0.73   0.1716 0.521  0.0543]
 [0.2    0.1573 0.7536 0.3724]
 [0.4364 0.8292 0.7044 0.2497]
 [0.4121 0.8949 0.612  0.2379]
 [0.9342 0.614  0.5356 0.5899]
 [1.0864 0.631  0.4417 0.5178]
 [0.2747 0.4748 0.4896 0.504 ]
 [0.2274 0.2544 0.058  0.4344]
 [0.3118 0.6575 0.3778 0.1796]
 [0.0247 0.0672 0.6583 0.4537]
 [0.5366 0.8967 0.9903 0.2169]
 [0.7139 0.2633 0.0207 0.8476]
 [0.32   0.3835 0.5883 0.831 ]
 [0.7608 1.3898 0.6772 0.798 ]
 [0.3321 0.5003 0.423  0.2845]
 [0.5159 0.4794 0.3253 0.2545]
 [0.4841 0.0257 0.2649 0.4273]
 [0.3742 0.4636 0.2776 0.5868]
 [0.8639 0.1175 0.5174 0.1321]
 [0.7169 0.3961 0.5654 0.1833]
 [0.1448 0.4881 0.3556 0.9404]
 [0.7653 0.7487 0.9037 0.0834]]
```

## Auteur

Ce projet fait partie du curriculum Machine Learning de l'École Holberton.

## Dépôt

- **Dépôt GitHub**: holbertonschool-machine_learning
- **Répertoire**: reinforcement_learning/temporal_difference
