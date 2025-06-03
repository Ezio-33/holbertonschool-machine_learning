#!/usr/bin/env python3
"""
Module Dataset pour la préparation des données de traduction
"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Classe Dataset : charge et prépare les données pour la traduction
    portugais → anglais à l’aide de Transformers.
    """

    def __init__(self):
        """
        Constructeur de la classe Dataset.
        Il initialise :
        - le dataset d'entraînement (pt → en)
        - le dataset de validation
        - les tokenizers pour le portugais et l’anglais
        """
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )

        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Crée les tokenizers à partir du dataset d'entraînement.

        Args:
            data: tf.data.Dataset, paires (pt, en)

        Returns:
            tuple: (tokenizer_pt, tokenizer_en)
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        def get_pt_corpus():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def get_en_corpus():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            get_pt_corpus(), vocab_size=2**13
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            get_en_corpus(), vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
