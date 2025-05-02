# Projet GANs : Génération de Visages et Applications

## Description

Ce projet explore l'utilisation des réseaux générateurs antagonistes (GANs) pour générer des visages réalistes. Il inclut plusieurs implémentations de GANs, y compris des versions simples, des WGANs avec clipping des poids, et des WGANs avec pénalité de gradient (WGAN-GP). Le projet est conçu pour démontrer les concepts fondamentaux des GANs et leurs applications pratiques.

## Table des Matières

1. [Installation](#installation)
2. [Utilisation](#utilisation)
3. [Structure du Projet](#structure-du-projet)
4. [Implémentations](#implémentations)
5. [Résultats](#résultats)
6. [Contributions](#contributions)

## Installation

Pour exécuter ce projet, vous aurez besoin de Python 3.x et des bibliothèques suivantes :

- TensorFlow
- NumPy
- Matplotlib

Vous pouvez installer les dépendances nécessaires en utilisant `pip` :

```bash
pip install tensorflow numpy matplotlib
```

## Utilisation

Pour entraîner un modèle GAN et générer des visages, exécutez le script principal :

```bash
python 0-main.py
```

Vous pouvez également visualiser les résultats en utilisant les scripts de visualisation fournis.

## Structure du Projet

Le projet est organisé comme suit :

```
gan/
├── 0-main.py
├── 0-main_01.py
├── 0-main_02.py
├── 0-main_03.py
├── 0-main_20.py
├── 0-main_21.py
├── 0-simple_gan.py
├── 1-main.py
├── 1-wgan_clip.py
├── 2-wgan_gp.py
├── 3-generate_faces.py
├── 4-wgan_gp.py
├── README.md
└── small_res_faces_10000.npy
```

## Implémentations

### 0. Simple GAN

Le fichier `0-simple_gan.py` contient l'implémentation d'un GAN simple. Ce modèle utilise une fonction de perte basée sur la distance quadratique moyenne (MSE) pour entraîner le générateur et le discriminateur.

### 1. WGAN avec Clipping des Poids

Le fichier `1-wgan_clip.py` implémente un WGAN avec clipping des poids. Cette version utilise la perte de Wasserstein et applique un clipping des poids du discriminateur pour stabiliser l'entraînement.

### 2. WGAN avec Pénalité de Gradient

Le fichier `2-wgan_gp.py` contient l'implémentation d'un WGAN avec pénalité de gradient (WGAN-GP). Cette version utilise une pénalité de gradient pour encourager les gradients unitaires dans le discriminateur, ce qui améliore la stabilité de l'entraînement.

### 3. Génération de Visages

Le fichier `3-generate_faces.py` définit les architectures des générateurs et discriminateurs convolutionnels utilisés pour générer des visages. Ce script est utilisé pour entraîner des modèles GAN sur un ensemble de données de visages.

## Résultats

Les résultats de l'entraînement des modèles GAN sont visualisés à l'aide de scripts de visualisation. Par exemple, le fichier `0-main.py` génère des visages et affiche les résultats à l'aide de Matplotlib.

## Contributions

Les contributions à ce projet sont les bienvenues. Si vous souhaitez contribuer, veuillez ouvrir une issue ou soumettre une pull request.

`
