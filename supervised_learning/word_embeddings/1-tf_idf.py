#!/usr/bin/env python3
"""
Crée une matrice d'embedding avec la méthode TF-IDF.
"""

import numpy as np
import math
import re


def tf_idf(sentences, vocab=None):
    """
    Génère une matrice d'embedding selon la méthode TF-IDF.

    Args:
        sentences (list): Liste des phrases (strings)
        vocab (list, optional): Liste des mots à utiliser.
		Si None, on extrait tous les mots.

    Returns:
        tuple:
            embeddings (numpy.ndarray): Matrice TF-IDF (phrases x mots)
            features (list): Liste triée des mots utilisés
    """
    cleaned = [
        [re.sub(r"[.,!?]", "", w).lower() for w in s.split()]
        for s in sentences
    ]

    if vocab is None:
        features = sorted({word for sent in cleaned for word in sent})
    else:
        features = vocab

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f))

    for i, sent in enumerate(cleaned):
        for j, word in enumerate(features):
            tf = sent.count(word) / len(sent)

            doc_count = sum(1 for s in cleaned if word in s)
            if doc_count == 0:
                idf = 0
            else:
                idf = math.log(s / doc_count)

            embeddings[i][j] = tf * idf

    return embeddings, features
