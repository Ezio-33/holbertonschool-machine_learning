#!/usr/bin/env python3
"""
Ce module calcule le score BLEU cumulatif d'une phrase
en utilisant les n-grammes de 1 jusqu'à n.
"""

import numpy as np
from collections import Counter


def cumulative_bleu(references, sentence, n):
    """
    Calcule le score BLEU cumulatif basé sur les n-grammes jusqu’à n.

    Args:
        references (list of list of str): Phrases de référence.
        sentence (list of str): Phrase générée automatiquement.
        n (int): Plus grand n-gramme à utiliser.

    Returns:
        float: Score BLEU cumulatif.
    """
    precisions = []

    for i in range(1, n + 1):
        # Extraction des n-grammes
        ngrams_sentence = [tuple(sentence[j:j + i])
                           for j in range(len(sentence) - i + 1)]
        sentence_counter = Counter(ngrams_sentence)

        total_ngrams = sum(sentence_counter.values())
        if total_ngrams == 0:
            return 0

        # Compter les n-grammes dans les références
        max_ref_counter = Counter()
        for ref in references:
            ref_ngrams = [tuple(ref[j:j + i]) for j in range(len(ref) - i + 1)]
            ref_counter = Counter(ref_ngrams)
            for ngram in ref_counter:
                max_ref_counter[ngram] = max(
                    max_ref_counter[ngram], ref_counter[ngram])

        clipped_count = sum(min(count,
                                max_ref_counter[ngram]) for ngram,
                            count in sentence_counter.items())

        precision = clipped_count / total_ngrams
        if precision == 0:
            return 0
        precisions.append(np.log(precision))

    # Moyenne géométrique des log(precisions)
    geo_mean = np.exp(sum(precisions) / n)

    # Brevity Penalty
    closest_ref_len = len(
        min(references, key=lambda ref: abs(len(ref) - len(sentence))))
    if len(sentence) < closest_ref_len:
        BP = np.exp(1 - (closest_ref_len / len(sentence)))
    else:
        BP = 1

    return BP * geo_mean
