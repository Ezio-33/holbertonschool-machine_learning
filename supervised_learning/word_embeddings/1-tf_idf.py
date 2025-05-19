#!/usr/bin/env python3
"""
Crée une matrice d'embedding avec la méthode TF-IDF à l'aide de sklearn.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Génère une matrice TF-IDF à partir d'une liste de phrases.

    Args:
        sentences (list): Liste de chaînes de caractères (phrases).
        vocab (list): Liste optionnelle de mots (features) à utiliser.

    Returns:
        tuple:
            numpy.ndarray: matrice TF-IDF
            list: liste des features utilisées
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    tfidf_matrix = vectorizer.fit_transform(sentences).todense()
    features = vectorizer.get_feature_names_out()
    return tfidf_matrix, features
