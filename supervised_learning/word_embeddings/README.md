# 📚 NLP - Word Embeddings

## 🎯 Objectif du projet

Ce projet a été réalisé dans le cadre de ma spécialisation en Machine Learning chez **Holberton School**.

L’objectif est de découvrir et manipuler différentes méthodes de représentation vectorielle des mots (word embeddings) en **Natural Language Processing (NLP)**, avec notamment :

- Bag of Words
- TF-IDF
- Word2Vec
- FastText
- ELMo (question théorique)

---

## 🧠 Compétences visées

- Prétraitement de texte (normalisation, tokenisation)
- Construction de vocabulaire
- Transformation de phrases en vecteurs numériques
- Entraînement de modèles Word2Vec et FastText avec **Gensim**
- Intégration d'embeddings dans des réseaux de neurones avec **TensorFlow Keras**
- Compréhension des embeddings contextuels comme **ELMo**

---

## 📁 Contenu des fichiers

| Fichier                   | Description                                                     |
| ------------------------- | --------------------------------------------------------------- |
| `0-bag_of_words.py`       | Génère une matrice d'embedding Bag of Words                     |
| `1-tf_idf.py`             | Génère une matrice d'embedding avec TF-IDF (`sklearn`)          |
| `2-word2vec.py`           | Entraîne un modèle Word2Vec avec Gensim                         |
| `3-gensim_to_keras.py`    | Convertit un modèle Gensim Word2Vec en couche Keras `Embedding` |
| `4-fasttext.py`           | Entraîne un modèle FastText avec Gensim                         |
| `5-elmo`                  | Réponse théorique (texte brut) à la question sur ELMo           |
| `0-main.py` → `4-main.py` | Fichiers de test pour chaque task                               |
| `README.md`               | Ce fichier de documentation du projet                           |

---

## ⚙️ Technologies utilisées

- Python 3.9
- Numpy 1.25.2
- TensorFlow 2.15 / Keras 2.15
- Gensim 4.3.3
- Scikit-learn (`TfidfVectorizer`)
- Ubuntu 20.04 LTS

---

## ✅ Règles de style respectées

- Tous les fichiers commencent par `#!/usr/bin/env python3`
- Code conforme à **pycodestyle 2.11.1**
- Documentation complète :
  - du module
  - de chaque fonction
- Tous les fichiers se terminent par une **nouvelle ligne**

---

## ✍️ Auteur

Ce projet a été réalisé par **[Samuel VERSCHUEREN]**  
dans le cadre du cursus Machine Learning de **Holberton School**.

---

## 📌 Licence

Projet réalisé à des fins d'apprentissage.  
Pas de licence d’exploitation commerciale.
