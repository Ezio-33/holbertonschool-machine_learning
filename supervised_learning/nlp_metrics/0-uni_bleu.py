#!/usr/bin/env python3
"""
Ce module calcule le score BLEU unigramme pour une phrase donnée.
"""

import numpy as np
from collections import Counter


def uni_bleu(references, sentence):
    """
    Calcule le score BLEU unigramme pour une phrase donnée.

    Args:
        references (list of list of str): Traductions de référence.
        sentence (list of str): Phrase générée par le modèle.

    Returns:
        float: Score BLEU unigramme.
    """
    # Compter les mots de la phrase
    sentence_counts = Counter(sentence)

    # Compter le max de chaque mot parmi les références
    max_ref_counts = Counter()
    for ref in references:
        ref_counter = Counter(ref)
        for word in ref_counter:
            max_ref_counts[word] = max(max_ref_counts[word], ref_counter[word])

    # Clipping : limiter les doublons à ce que les références autorisent
    clipped_counts = {word: min(count, max_ref_counts[word])
                      for word, count in sentence_counts.items()}
    total_clipped = sum(clipped_counts.values())

    precision = total_clipped / len(sentence)

    # Brevity penalty
    r = find_closest(references, sentence)
    ref_len = len(references[r])

    if len(sentence) < ref_len:
        BP = np.exp(1 - (ref_len / len(sentence)))
    else:
        BP = 1

    return BP * precision


def find_closest(references, sentence):
    """
    Trouve la référence dont la longueur est la plus proche de la phrase.

    Args:
        references (list of list of str): Traductions de référence.
        sentence (list of str): Phrase générée par le modèle.

    Returns:
        int: Index de la référence la plus proche.
    """
    distances = [abs(len(ref) - len(sentence)) for ref in references]
    return distances.index(min(distances))
