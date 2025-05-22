#!/usr/bin/env python3
"""
Ce module calcule le score BLEU n-gramme pour une phrase.
"""

import numpy as np
from collections import Counter


def ngram_bleu(references, sentence, n):
    """
    Calcule le score BLEU basé sur les n-grammes.

    Args:
        references (list of list of str): Phrases de référence.
        sentence (list of str): Phrase générée par le modèle.
        n (int): Taille des n-grammes à utiliser.

    Returns:
        float: Score BLEU n-gramme.
    """
    if len(sentence) < n:
        return 0

    # Générer les n-grammes pour la phrase
    ngrams_in_sentence = [tuple(sentence[i:i + n])
                          for i in range(len(sentence) - n + 1)]
    candidate_counts = Counter(ngrams_in_sentence)
    total_ngrams = sum(candidate_counts.values())

    # Compter les n-grammes dans les références (avec max entre références)
    reference_counts = Counter()
    for ref in references:
        ref_ngrams = [tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)]
        for ngram in set(ref_ngrams):
            reference_counts[ngram] = max(
                reference_counts[ngram], ref_ngrams.count(ngram))

    # Clipping : ne pas dépasser la fréquence max trouvée en référence
    clipped_counts = {ngram: min(count, reference_counts[ngram])
                      for ngram, count in candidate_counts.items()}
    matched_ngrams = sum(clipped_counts.values())
    precision = matched_ngrams / total_ngrams

    # Brevity Penalty
    r = find_closest(references, sentence)
    ref_len = len(references[r])
    if len(sentence) < ref_len:
        BP = np.exp(1 - (ref_len / len(sentence)))
    else:
        BP = 1

    return precision * BP


def find_closest(references, sentence):
    """
    Trouve la référence dont la longueur est la plus proche de la phrase.

    Args:
        references (list of list of str): Phrases de référence.
        sentence (list of str): Phrase générée.

    Returns:
        int: Index de la meilleure correspondance.
    """
    distances = [abs(len(ref) - len(sentence)) for ref in references]
    return distances.index(min(distances))
