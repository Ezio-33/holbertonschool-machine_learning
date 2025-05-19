#!/usr/bin/env python3
"""
Crée, construit et entraîne un modèle Word2Vec avec Gensim.
"""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    Entraîne un modèle Word2Vec sur une liste de phrases.

    Args:
        sentences (list): Liste de phrases (liste de listes de mots).
        vector_size (int): Dimension de l’espace d’embedding.
        min_count (int): Mot ignoré s’il apparaît moins que min_count.
        window (int): Taille de la fenêtre de contexte.
        negative (int): Nombre de mots pour l’échantillonnage négatif.
        cbow (bool): True = CBOW, False = Skip-Gram.
        epochs (int): Nombre d’itérations d’entraînement.
        seed (int): Graine aléatoire.
        workers (int): Threads d’entraînement en parallèle.

    Returns:
        gensim.models.Word2Vec: Le modèle entraîné
    """
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=not cbow,
        seed=seed,
        workers=workers
    )

    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model
