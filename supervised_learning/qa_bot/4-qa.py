#!/usr/bin/env python3
"""
Module Q/R avancé avec recherche sémantique multi-documents

Ce module permet de répondre à des questions posées en langage naturel
en identifiant automatiquement le document le plus pertinent dans un corpus,
et en extrayant une réponse grâce au modèle BERT.
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from transformers import BertTokenizer


# Chargement global du modèle de similarité sémantique
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
semantic_model = hub.load(module_url)

# Chargement du tokenizer et du modèle BERT pour la Q/R
tz = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad'
)
model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')


def question_answer(coprus_path):
    """
    Lance une session interactive de Q/R à partir d'un corpus de documents.

    Args:
        coprus_path (str): Le chemin du dossier contenant les fichiers `.md`.
    """
    while True:
        val = input("Q: ")
        exit_list = ['exit', 'quit', 'goodbye', 'bye']
        if val.lower() in exit_list:
            print("A: Goodbye")
            break

        reference = semantic_search(coprus_path, val)
        answer = q_answer(val, reference)

        if answer is None or answer == "":
            answer = "Sorry, I do not understand your question."

        print("A: {}".format(answer))


def semantic_search(corpus_path, sentence):
    """
    Recherche sémantique dans le corpus pour trouver le document
    le plus pertinent.

    Args:
        corpus_path (str): Dossier contenant les fichiers `.md`.
        sentence (str): La phrase ou question à analyser.

    Returns:
        str: Le texte du document le plus similaire à la question.
    """
    documents = [sentence]

    for filename in os.listdir(corpus_path):
        if not filename.endswith(".md"):
            continue
        file_path = os.path.join(corpus_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            documents.append(f.read())

    embed = semantic_model(documents)
    corr = np.inner(embed, embed)
    close = np.argmax(corr[0, 1:])
    similarity = documents[close + 1]

    return similarity


def q_answer(question, reference):
    """
    Utilise BERT pour trouver une réponse dans un document donné.

    Args:
        question (str): La question posée.
        reference (str): Le texte dans lequel chercher la réponse.

    Returns:
        str: Une réponse extraite du texte, ou None si non trouvée.
    """
    question_ts = tz.tokenize(question)
    paragraph_ts = tz.tokenize(reference)
    tokens = ['[CLS]'] + question_ts + ['[SEP]'] + paragraph_ts + ['[SEP]']
    word_ids = tz.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(word_ids)
    type_ids = [0] * (1 + len(question_ts) + 1) + [1] * (len(paragraph_ts) + 1)

    word_ids, input_mask, type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (word_ids, input_mask, type_ids))

    outputs = model([word_ids, input_mask, type_ids])
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tz.convert_tokens_to_string(answer_tokens)

    return answer
