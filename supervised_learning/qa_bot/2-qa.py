#!/usr/bin/env python3
"""
Q/A ChatBot module
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

# Initialisations globales
tz = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad'
)
model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')


def question_answer(question, reference):
    """Trouve un extrait de texte dans un document pour répondre à une
    question"""
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

    # Extraction des indices de début/fin de la réponse
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tz.convert_tokens_to_string(answer_tokens)
    return answer


def answer_loop(reference):
    """Boucle interactive de questions/réponses depuis un texte de référence"""
    while True:
        val = input("Q: ")
        exit_list = ['exit', 'quit', 'goodbye', 'bye']
        if val.lower() in exit_list:
            print("A: Goodbye")
            break
        answer = question_answer(val, reference)
        if answer is None or answer == "":
            answer = "Sorry, I do not understand your question."
        print("A: {}".format(answer))
