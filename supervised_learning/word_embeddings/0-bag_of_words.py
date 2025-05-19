#!/usr/bin/env python3
"""
Module qui crée une représentation vectorielle des phrases avec Bag of Words.
"""

import numpy as np
import string


def bag_of_words(sentences, vocab=None):
    """
    Crée une matrice d'embedding selon la méthode Bag of Words.

    Args:
        sentences (list): Liste de phrases (chaînes de caractères).
        vocab (list, optional): Liste de mots à utiliser comme vocabulaire.
            Si None, tous les mots des phrases seront utilisés.

    Returns:
        tuple:
            embeddings (numpy.ndarray): Matrice (phrases x mots)
            features (list): Liste triée des mots utilisés
    """
    # Nettoyage : suppression ponctuation + passage en minuscules
    table = str.maketrans('', '', string.punctuation)
    tokenized = [
        sentence.lower().translate(table).split() for sentence in sentences
    ]

    if vocab is None:
        vocab_set = set()
        for sent in tokenized:
            vocab_set.update(sent)
        vocab = sorted(vocab_set)

    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    # Initialisation de la matrice
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, sent in enumerate(tokenized):
        for word in sent:
            if word in word_to_index:
                embeddings[i][word_to_index[word]] += 1

    return embeddings, vocab
