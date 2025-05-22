#!/usr/bin/env python3
"""
Ce module calcule le score BLEU unigramme pour une phrase donnée.
"""

import numpy as np


def uni_bleu(references, sentence):
    """
    Calcule le score BLEU unigramme pour une phrase donnée.

    Args:
        references (list of list of str): Traductions de référence.
        sentence (list of str): Phrase générée par le modèle.

    Returns:
        float: Score BLEU unigramme.
    """
    count = 0
    for word in sentence:
        refs = [ref.count(word) for ref in references]
        count += min(sentence.count(word), max(refs))

    # Précision P1 = nb mots corrects / nb total mots dans la phrase générée
    P1 = count / len(sentence)

    # Trouver la référence la plus proche en longueur
    r = find_closest(references, sentence)
    ref_closest_len = len(references[r])

    # Brevity Penalty (BP)
    if len(sentence) < ref_closest_len:
        BP = np.exp(1 - (ref_closest_len / len(sentence)))
    else:
        BP = 1

    return P1 * BP


def find_closest(references, sentence):
    """
    Trouve la référence dont la longueur est la plus proche de la phrase générée.

    Args:
        references (list of list of str): Traductions de référence.
        sentence (list of str): Phrase générée par le modèle.

    Returns:
        int: Index de la référence la plus proche.
    """
    distances = [abs(len(ref) - len(sentence)) for ref in references]
    return distances.index(min(distances))
