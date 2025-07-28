# Pandas - Data Manipulation and Analysis

## Description

Ce projet fait partie du cursus de formation en Machine Learning de l'école Holberton. Il se concentre sur l'apprentissage et la maîtrise de la bibliothèque **pandas**, un outil essentiel pour la manipulation et l'analyse de données en Python.

**Pandas** est une bibliothèque Python puissante qui fournit des structures de données flexibles et des outils d'analyse de données. Elle est particulièrement utile pour :

- La manipulation de données tabulaires (DataFrames)
- Le nettoyage et la préparation des données
- L'analyse exploratoire des données
- La visualisation de données

## Objectifs d'apprentissage

À la fin de ce projet, vous devriez être capable d'expliquer sans aide externe :

### Concepts généraux

- **Qu'est-ce que pandas ?** - Une bibliothèque Python pour l'analyse et la manipulation de données
- **Qu'est-ce qu'un pd.DataFrame ?** - Structure de données tabulaire 2D avec axes étiquetés
- **Qu'est-ce qu'une pd.Series ?** - Structure de données 1D étiquetée
- **Comment charger des données depuis un fichier** (CSV, Excel, etc.)
- **Comment effectuer l'indexation sur un pd.DataFrame**
- **Comment utiliser l'indexation hiérarchique avec un pd.DataFrame**
- **Comment découper un pd.DataFrame** (slicing)
- **Comment réassigner des colonnes**
- **Comment trier un pd.DataFrame**
- **Comment utiliser la logique booléenne avec un pd.DataFrame**
- **Comment fusionner/concaténer/joindre des pd.DataFrames**
- **Comment obtenir des informations statistiques d'un pd.DataFrame**
- **Comment visualiser un pd.DataFrame**

## Prérequis techniques

- **Système** : Ubuntu 20.04 LTS
- **Python** : Version 3.9
- **Bibliothèques** :
  - numpy (version 1.25.2)
  - pandas (version 2.2.2)
- **Style de code** : pycodestyle (version 2.11.1)

## Installation

```bash
pip install --user pandas==2.2.2
```

## Jeux de données

Ce projet utilise les jeux de données de **coinbase** et **bitstamp**, des données de séries temporelles financières utilisées précédemment dans les projets de prévision de séries temporelles.

## Structure du projet

### Tâches implémentées

#### 0. From Numpy (`0-from_numpy.py`)

**Objectif** : Créer un DataFrame pandas à partir d'un tableau numpy

- Convertit un `np.ndarray` en `pd.DataFrame`
- Étiquette automatiquement les colonnes alphabétiquement (A, B, C, ...)
- Gère jusqu'à 26 colonnes

#### 1. From Dictionary (`1-from_dictionary.py`)

**Objectif** : Créer un DataFrame pandas à partir d'un dictionnaire

- Crée un DataFrame avec des colonnes 'First' et 'Second'
- Utilise des index personnalisés (A, B, C, D)
- Démontre la création de DataFrames avec des données mixtes

#### 2. From File (`2-from_file.py`)

**Objectif** : Charger des données depuis un fichier

- Lit des fichiers CSV avec délimiteurs personnalisés
- Retourne un DataFrame pandas prêt à l'emploi
- Gère différents formats de fichiers

#### 3. Rename (`3-rename.py`)

**Objectif** : Renommer et transformer des colonnes

- Renomme la colonne 'Timestamp' en 'Datetime'
- Convertit les timestamps en objets datetime
- Filtre pour ne garder que les colonnes Datetime et Close

#### 4. To Numpy (`4-array.py`)

**Objectif** : Convertir un DataFrame en tableau numpy

- Sélectionne les 10 dernières lignes des colonnes High et Close
- Convertit en `numpy.ndarray`
- Utile pour l'interfaçage avec d'autres bibliothèques

#### 5. Slice (`5-slice.py`)

**Objectif** : Découpage avancé de DataFrames

- Extrait des colonnes spécifiques (High, Low, Close, Volume\_(BTC))
- Sélectionne chaque 60ème ligne
- Démontre les techniques de sous-échantillonnage

#### 6. Flip it and Switch it (`6-flip_switch.py`)

**Objectif** : Transformation et transposition

- Trie en ordre chronologique inverse
- Transpose le DataFrame (lignes ↔ colonnes)
- Útile pour la réorganisation des données

#### 7. Sort (`7-high.py`)

**Objectif** : Tri de données

- Trie par prix maximum (High) en ordre décroissant
- Démontre les techniques de tri pandas
- Identifie les valeurs extrêmes

#### 8. Prune (`8-prune.py`)

**Objectif** : Nettoyage des données

- Supprime les lignes avec des valeurs NaN dans Close
- Pratique essentielle de nettoyage de données
- Améliore la qualité des données

#### 9. Fill (`9-fill.py`)

**Objectif** : Gestion avancée des valeurs manquantes

- Supprime la colonne Weighted_Price
- Remplit les valeurs manquantes avec différentes stratégies :
  - Close : valeur de la ligne précédente
  - High, Low, Open : valeur Close de la même ligne
  - Volume : 0

#### 10. Indexing (`10-index.py`)

**Objectif** : Indexation personnalisée

- Définit Timestamp comme index du DataFrame
- Améliore les performances de recherche
- Facilite les opérations temporelles

#### 11. Concat (`11-concat.py`)

**Objectif** : Concaténation de DataFrames

- Combine deux DataFrames (coinbase et bitstamp)
- Applique un filtrage temporel
- Ajoute des labels hiérarchiques

#### 12. Hierarchy (`12-hierarchy.py`)

**Objectif** : Index hiérarchique (MultiIndex)

- Réorganise les index pour Timestamp en premier niveau
- Concatène des données dans une plage temporelle spécifique
- Maintient l'ordre chronologique

#### 13. Analyze (`13-analyze.py`)

**Objectif** : Analyse statistique

- Calcule des statistiques descriptives
- Exclut les colonnes non numériques
- Fournit des insights sur la distribution des données

#### 14. Visualize (`14-visualize.py`)

**Objectif** : Visualisation et agrégation

- Prépare les données pour la visualisation
- Agrège par jour avec différentes méthodes :
  - High : maximum
  - Low : minimum
  - Open/Close : moyenne
  - Volume : somme
- Génère des graphiques pour l'analyse temporelle

## Compétences développées

### Manipulation de données

- **Chargement** : CSV, dictionnaires, arrays numpy
- **Nettoyage** : Gestion des valeurs manquantes, suppression de lignes/colonnes
- **Transformation** : Renommage, conversion de types, agrégation

### Analyse de données

- **Statistiques descriptives** : mean, std, min, max, quartiles
- **Filtrage** : Sélection conditionnelle de données
- **Tri et indexation** : Organisation optimale des données

### Opérations avancées

- **Concaténation et fusion** : Combinaison de datasets multiples
- **Index hiérarchique** : Gestion de données multi-dimensionnelles
- **Agrégation temporelle** : Regroupement par périodes

### Visualisation

- **Préparation des données** pour matplotlib
- **Création de graphiques** temporels
- **Analyse de tendances** financières

## Contexte professionnel

Ce projet simule des scenarios réels de data science :

- **Analyse financière** avec des données de trading
- **Nettoyage de datasets** réels avec valeurs manquantes
- **Préparation de données** pour le machine learning
- **Création de rapports** et visualisations

## Évaluation

Le projet comprend 14 tâches obligatoires évaluées sur :

- **Fonctionnalité** : Respect des spécifications
- **Qualité du code** : Style pycodestyle, documentation
- **Bonnes pratiques** : Gestion d'erreurs, efficacité

**Note** : Ce projet nécessite une revue manuelle QA à la fin.

---

_Ce projet fait partie du programme de spécialisation en Machine Learning de Holberton School, conçu pour former les futurs data scientists et ingénieurs ML aux outils essentiels de l'industrie._
