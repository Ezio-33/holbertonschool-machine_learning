#!/usr/bin/env python3
"""
Module d'entraînement d'un modèle Transformer pour la traduction
"""

import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Planification personnalisée du taux d'apprentissage (Warmup + decay)
    """

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * tf.math.pow(self.warmup_steps, -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Entraîne un modèle Transformer pour la traduction pt → en.

    Args:
        N (int): nombre de blocs dans l'encodeur/décodeur
        dm (int): dimension du modèle
        h (int): nombre de têtes d'attention
        hidden (int): taille de la couche feedforward
        max_len (int): longueur maximale de séquence
        batch_size (int): taille des batchs
        epochs (int): nombre d'époques d'entraînement

    Returns:
        Transformer: modèle entraîné
    """
    data = Dataset(batch_size, max_len)
    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(N, dm, h, hidden,
                              input_vocab, target_vocab,
                              max_len, max_len)

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )

    def loss_function(y_true, y_pred):
        """
        Calcule la perte avec masquage du padding (valeurs = 0).
        """
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        loss = loss_object(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="train_accuracy")

    for epoch in range(epochs):
        batch = 0
        for inputs, targets in data.data_train:
            target_input = targets[:, :-1]
            target_real = targets[:, 1:]

            enc_mask, combined_mask, dec_mask = create_masks(
                inputs, target_input)

            with tf.GradientTape() as tape:
                predictions = transformer(inputs, target_input, True,
                                          enc_mask, combined_mask, dec_mask)
                loss = loss_function(target_real, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, transformer.trainable_variables))

            train_loss(loss)
            train_accuracy(target_real, predictions)

            if batch % 50 == 0:
                print("Epoch {}, batch {}: loss {:.4f} accuracy {:.4f}".format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            batch += 1

        print("Epoch {}: loss {:.4f} accuracy {:.4f}".format(
            epoch + 1, train_loss.result(), train_accuracy.result()))

        train_loss.reset_states()
        train_accuracy.reset_states()

    return transformer
