#!/usr/bin/env python3
"""
Module pour calculer les encodages positionnels dans un Transformer
"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calcule l'encodage positionnel pour un Transformer.

    Args:
        max_seq_len (int): longueur maximale de la séquence.
        dm (int): profondeur du modèle (dimension des embeddings).

    Returns:
        np.ndarray: tableau de forme (max_seq_len, dm)
                    contenant les vecteurs d'encodage positionnel.
    """
    PE = np.zeros((max_seq_len, dm))

    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            angle = pos / np.power(10000, (2 * (i // 2)) / dm)
            PE[pos, i] = np.sin(angle)
            if i + 1 < dm:
                PE[pos, i + 1] = np.cos(angle)

    return PE
