# Transfer Learning

## Description

Ce projet montre comment entraîner un modèle de transfert learning pour classifier des images du dataset CIFAR10. Il met en avant l'utilisation d'un modèle pré-entraîné (Xception) et explique pourquoi le gel et le dégèlement des couches sont essentiels pour un entraînement efficace.

## Definitions

- **Transfer learning** : technique utilisant un modèle pré-entraîné sur un large jeu de données pour une nouvelle tâche.
- **Fine-tuning** : ajuster certaines couches (souvent supérieures) du réseau pré-entraîné pour améliorer les performances.
- **Frozen layer** : couche dont les poids ne sont pas mis à jour durant l’entraînement. On gèle généralement les couches pour tirer parti des fonctionnalités apprises sur d’autres données.

## Learning Objectives

- Comprendre le principe du transfert learning
- Savoir geler et dégeler des couches
- Utiliser des modèles Keras pré-entraînés
- Prétraiter correctement des images pour aligner les dimensions
- Atteindre une précision de validation d’au moins 87%

## Requirements

- Python 3.9, TensorFlow 2.15, NumPy 1.25.2
- Pycodestyle 2.11.1
- Toute fonction, classe ou module doit être documenté
- Fichiers exécutables sous Ubuntu 20.04 LTS

## Tasks

1. **Transfer Knowledge**
   - Utiliser un modèle de Keras Applications avec un layer Lambda de redimensionnement.
   - Geler la majorité des couches, puis affiner les couches finales.
   - Enregistrer le modèle dans un fichier `cifar10.h5` avec une précision de validation >= 87%.
2. **Rédaction d’un blog**
   - Expliquer le processus expérimental (introduction, méthodes, résultats, etc.)
