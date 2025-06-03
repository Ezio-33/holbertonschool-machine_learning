#!/usr/bin/env python3
"""
Module Dataset : Encodage TensorFlow compatible pour traduction automatique
"""

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
    Classe Dataset : prépare le jeu de données et encode les phrases
    en tensors pour entraînement Transformer.
    """

    def __init__(self):
        """
        Constructeur :
        - Charge les données
        - Crée les tokenizers
        - Applique tf_encode() automatiquement aux paires (pt, en)
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

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Crée les tokenizers à partir du dataset d'entraînement.

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
        Encode une paire (pt, en) en entiers (tokens)
        avec start/end personnalisés.

        Args:
            pt: tf.Tensor ou str
            en: tf.Tensor ou str

        Returns:
            (pt_tokens, en_tokens) : list[int]
        """
        if tf.is_tensor(pt):
            pt = pt.numpy().decode('utf-8')
        if tf.is_tensor(en):
            en = en.numpy().decode('utf-8')

        pt_start = self.tokenizer_pt.vocab_size
        pt_end = pt_start + 1
        en_start = self.tokenizer_en.vocab_size
        en_end = en_start + 1

        pt_tokens = [pt_start] + \
            self.tokenizer_pt.encode(pt, add_special_tokens=False) + \
            [pt_end]

        en_tokens = [en_start] + \
            self.tokenizer_en.encode(en, add_special_tokens=False) + \
            [en_end]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Encapsule encode() dans une fonction TensorFlow-compatible
        pour l'appliquer via tf.data.Dataset.map()

        Args:
            pt: tf.Tensor
            en: tf.Tensor

        Returns:
            (pt_tensor, en_tensor) : tf.Tensor
        """
        tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_tensor = tf.ensure_shape(tokens[0], [None])
        en_tensor = tf.ensure_shape(tokens[1], [None])

        return pt_tensor, en_tensor
