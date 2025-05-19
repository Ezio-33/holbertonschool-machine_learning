#!/usr/bin/env python3
"""
Module qui crée une représentation vectorielle des phrases en utilisant la méthode Bag of Words.
"""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Crée une matrice d'embedding avec la méthode Bag of Words.

    Args:
        sentences (list): Liste de phrases (chaînes de caractères).
        vocab (list, optional): Liste de mots à utiliser comme vocabulaire.
            Si None, tous les mots des phrases seront utilisés.

    Returns:
        tuple:
            embeddings (numpy.ndarray): Matrice des embeddings (phrases x mots)
            features (list): Liste des mots/features utilisés
    """
    # Nettoyage et découpe des phrases
    tokenized_sentences = [
        re.findall(r'\b\w+\b', sentence.lower()) for sentence in sentences
    ]

    # Création du vocabulaire si non fourni
    if vocab is None:
        vocab_set = set()
        for sentence in tokenized_sentences:
            vocab_set.update(sentence)
        vocab = sorted(vocab_set)

    # Création d’un mapping mot → index
    word_index = {word: idx for idx, word in enumerate(vocab)}

    # Initialisation de la matrice d’embedding
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    # Remplissage de la matrice
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in word_index:
                embeddings[i][word_index[word]] += 1

    return embeddings, vocab
