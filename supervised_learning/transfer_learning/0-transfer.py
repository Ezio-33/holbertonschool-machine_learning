#!/usr/bin/env python3
"""Modèle de transfert learning avec Xception pour CIFAR10"""
from tensorflow import keras as K
import tensorflow as tf

def preprocess_data(X, Y):
    """
    Prétraitement des données pour Xception
    
    Args:
        X: numpy.ndarray de forme (m, 32, 32, 3)
        Y: numpy.ndarray de forme (m,)
    
    Returns:
        X_p, Y_p: Données prétraitées
    """
    X_p = K.applications.xception.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == '__main__':
    # Chargement et prétraitement des données
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Data Augmentation
    train_datagen = K.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.3
    )
    train_gen = train_datagen.flow(x_train, y_train, batch_size=128)

    # Architecture Xception
    inputs = K.Input(shape=(32, 32, 3))
    x = K.layers.Lambda(
        lambda img: tf.image.resize(img, (299, 299)), 
        name='resize_input'
    )(inputs)
    
    base_model = K.applications.Xception(
        include_top=False,
        weights='imagenet',
        input_tensor=x,
        pooling='avg'
    )
    base_model.trainable = False  # Gel des couches

    # Tête de classification personnalisée
    x = K.layers.Dense(1024, activation='relu')(base_model.output)
    x = K.layers.Dropout(0.4)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs, outputs)

    # Compilation avec learning
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    # Callbacks
    callbacks = [
        K.callbacks.EarlyStopping(patience=7, monitor='val_acc'),
        K.callbacks.ModelCheckpoint('cifar10.h5', save_best_only=True),
        K.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
    ]

    # Entraînement
    history = model.fit(
        train_gen,
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
