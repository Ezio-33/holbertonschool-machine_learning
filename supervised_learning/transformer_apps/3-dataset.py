#!/usr/bin/env python3
"""
Module Dataset : pipeline d'entraînement optimisé pour Transformer
"""

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
    Classe Dataset : crée un pipeline de données adapté pour l'entraînement
    d’un modèle Transformer sur un jeu de données de traduction pt → en.
    """

    def __init__(self, batch_size, max_len):
        """
        Initialise les données et construit les pipelines optimisés
        Args:
            batch_size (int): taille des batchs
            max_len (int): longueur maximale de token par phrase
        """
        self.batch_size = batch_size
        self.max_len = max_len

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

        def filter_max_length(pt, en):
            """Filtre les phrases trop longues"""
            return tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )

        # Pipeline pour entraînement
        self.data_train = self.data_train.filter(filter_max_length)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(self.batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        # Pipeline pour validation (plus léger)
        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(self.batch_size)

    def tokenize_dataset(self, data):
        """
        Crée les tokenizers portugais et anglais à partir du dataset

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
        Encode une paire de phrases pt/en avec
        tokens de début/fin personnalisés

        Args:
            pt: tf.Tensor (phrase en portugais)
            en: tf.Tensor (phrase en anglais)

        Returns:
            tuple (pt_tokens, en_tokens) : listes d'entiers
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
        Fonction utilisée avec .map() pour encoder en Tensors compatibles

        Args:
            pt: tf.Tensor
            en: tf.Tensor

        Returns:
            (pt_tensor, en_tensor): tf.Tensor 1D
        """
        result = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_tensor = tf.ensure_shape(result[0], [None])
        en_tensor = tf.ensure_shape(result[1], [None])

        return pt_tensor, en_tensor
