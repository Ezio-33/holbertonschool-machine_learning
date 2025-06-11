#!/usr/bin/env python3
"""
Module de recherche sémantique pour ChatBot Q/R

Cette fonction permet d'analyser plusieurs documents
et de trouver celui qui est le plus pertinent
en fonction d'une question donnée.
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def semantic_search(corpus_path, sentence):
    """
    Effectue une recherche sémantique sur un corpus de documents.

    Args:
        corpus_path (str): Dossier contenant les fichiers de référence (.md).
        sentence (str): La phrase utilisée pour effectuer la recherche.

    Returns:
        str: Le document le plus similaire à la phrase.
    """
    # Liste de tous les documents, en commençant par la phrase à comparer
    documents = [sentence]

    # Lecture de chaque fichier .md du dossier
    for filename in os.listdir(corpus_path):
        if not filename.endswith(".md"):
            continue
        file_path = os.path.join(corpus_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            documents.append(f.read())

    # Chargement du modèle Universal Sentence Encoder
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = hub.load(module_url)

    # Calcul des vecteurs d'embedding pour tous les documents
    embed = model(documents)

    # Calcul de la similarité entre la phrase (documents[0]) et les autres
    corr = np.inner(embed, embed)

    """On prend la ligne 0 (phrase de départ) et on trouve l'index max
    (sans compter elle-même) Décale de +1 ensuite
    pour retrouver le vrai document"""
    close = np.argmax(corr[0, 1:])

    similarity = documents[close + 1]
    return similarity
