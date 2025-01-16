#!/usr/bin/env python3
"""
Module pour l'entraînement d'un modèle avec learning rate decay
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Entraîne un modèle avec learning rate decay

    Arguments:
        network: modèle à entraîner
        data: données d'entrée
        labels: étiquettes
        batch_size: taille des lots
        epochs: nombre d'époques
        validation_data: données de validation
        early_stopping: activation de l'arrêt précoce
        patience: patience pour l'arrêt précoce
        learning_rate_decay: activation de la décroissance du taux
        alpha: taux d'apprentissage initial
        decay_rate: taux de décroissance
        verbose: affichage des détails
        shuffle: mélange des données

    Returns:
        History de l'entraînement
    """
    callbacks = []

    if validation_data and early_stopping:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop)

    if validation_data and learning_rate_decay:
        def scheduler(epoch):
            """Calcule le taux d'apprentissage décroissant"""
            return alpha / (1 + decay_rate * epoch)

        lr_decay = K.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1
        )
        callbacks.append(lr_decay)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )

    return history
