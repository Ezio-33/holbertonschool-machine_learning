# NLP - Evaluation Metrics

## 📚 Description

Ce projet a été réalisé dans le cadre de la formation de spécialisation Machine Learning chez Holberton School Bordeaux.

L’objectif est d’implémenter différentes fonctions permettant d’évaluer automatiquement la qualité d’une phrase générée par un modèle de traitement automatique du langage naturel (NLP), en la comparant à des phrases de référence.  
Les métriques utilisées ici sont basées sur le **BLEU score**, un indicateur très courant notamment dans la traduction automatique.

Le projet se concentre sur les niveaux suivants :

- BLEU score pour les unigrams (mots seuls)
- BLEU score pour les n-grammes (groupes de `n` mots)
- BLEU score cumulatif sur plusieurs niveaux de n-grammes

---

## 🎯 Objectifs pédagogiques

À la fin de ce projet, je suis capable de :

- Comprendre ce qu’est un **n-gramme**
- Calculer des **précisions** basées sur les n-grammes
- Appliquer la **brevity penalty** pour éviter les tricheurs
- Calculer un **score BLEU cumulatif** robuste et équilibré

---

## 🛠️ Technologies utilisées

- Python 3.9
- Numpy 1.25.2
- Pas d’utilisation du module `nltk`
- Conforme au style `pycodestyle` 2.11.1

---

## 🗂️ Fichiers

| Fichier                | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| `0-uni_bleu.py`        | Calcule le score BLEU basé sur les unigrams               |
| `1-ngram_bleu.py`      | Calcule le score BLEU pour des n-grammes (avec `n` donné) |
| `2-cumulative_bleu.py` | Calcule le score BLEU cumulatif de 1 à `n`                |
| `0-main.py`            | Script de test pour le fichier `0-uni_bleu.py`            |
| `1-main.py`            | Script de test pour `1-ngram_bleu.py`                     |
| `2-main.py`            | Script de test pour `2-cumulative_bleu.py`                |

---

## ✅ Pré-requis Holberton

- Tous les fichiers commencent par `#!/usr/bin/env python3`
- Tous les fichiers sont **exécutables**
- Tous les fichiers sont conformes à **`pycodestyle` 2.11.1**
- Chaque module, classe et fonction est correctement **documenté**
- Aucun fichier ne contient de dépendance à `nltk`

---

## ✍️ Auteur

Ce projet a été réalisé par **[Samuel VERSCHUEREN]**  
dans le cadre du cursus Machine Learning de **Holberton School**.

---

## 📌 Licence

Projet réalisé à des fins d'apprentissage.  
Pas de licence d’exploitation commerciale.
