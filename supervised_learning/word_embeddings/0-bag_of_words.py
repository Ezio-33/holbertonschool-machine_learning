#!/usr/bin/env python3
"""
Crée une matrice d'embedding avec la méthode Bag of Words.
"""

import re
import numpy as np


def formatter(word):
    """
    Nettoie un mot en supprimant les possessifs ('s) et ponctuations finales.

    Args:
        word (str): Mot brut à formater

    Returns:
        str: Mot nettoyé
    """
    # Supprime 's (possessif)
    word = re.sub(r"(\w+)'s\b", r"\1", word)
    # Supprime la ponctuation restante
    word = re.sub(r"[.,!?]", "", word)
    return word


def compter(word, phrase):
    """
    Compte les occurrences exactes d'un mot dans une phrase.

    Args:
        word (str): Mot à chercher
        phrase (str): Phrase cible

    Returns:
        int: Nombre d'occurrences du mot
    """
    pattern = r'\b' + re.escape(word) + r'\b'
    return len(re.findall(pattern, phrase, re.IGNORECASE))


def bag_of_words(sentences, vocab=None):
    """
    Crée une matrice d'embedding selon la méthode Bag of Words.

    Args:
        sentences (list): Liste de phrases
        vocab (list): Vocabulaire (si None, sera déduit)

    Returns:
        tuple: (matrice embeddings, liste des features)
    """
    if vocab is None:
        features = set()
        for sentence in sentences:
            words = sentence.lower().split()
            for word in words:
                clean_word = formatter(word)
                features.add(clean_word)
        features = sorted(features)
    else:
        features = vocab

    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, sentence in enumerate(sentences):
        sentence = sentence.lower()
        for j, word in enumerate(features):
            embeddings[i, j] = compter(word, sentence)

    return embeddings, np.array(features)
