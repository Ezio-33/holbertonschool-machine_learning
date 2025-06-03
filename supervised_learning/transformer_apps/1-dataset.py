#!/usr/bin/env python3
"""
Module Dataset pour encoder les paires de phrases portugais-anglais
"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Classe Dataset : prépare les données et encode les phrases
    """

    def __init__(self):
        """
        Constructeur : charge les données et entraîne les tokenizers
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
        Crée les tokenizers depuis le dataset d'entraînement

        Args:
            data: tf.data.Dataset

        Returns:
            (tokenizer_pt, tokenizer_en)
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        def get_pt():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def get_en():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            get_pt(), vocab_size=2**13
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            get_en(), vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encode une paire (pt, en) en tokens avec start/end personnalisés

        Args:
            pt: tf.Tensor contenant la phrase portugaise
            en: tf.Tensor contenant la phrase anglaise

        Returns:
            tuple (pt_tokens, en_tokens) : listes numpy d'entiers
        """
        pt_start = self.tokenizer_pt.vocab_size
        pt_end = pt_start + 1
        en_start = self.tokenizer_en.vocab_size
        en_end = en_start + 1

        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_tokens = [pt_start] + \
            self.tokenizer_pt.encode(pt_text, add_special_tokens=False) + \
            [pt_end]

        en_tokens = [en_start] + \
            self.tokenizer_en.encode(en_text, add_special_tokens=False) + \
            [en_end]

        return pt_tokens, en_tokens
