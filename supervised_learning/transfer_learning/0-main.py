#!/usr/bin/env python3

from tensorflow import keras as K
import tensorflow as tf

# Désactiver l'utilisation du GPU
tf.config.set_visible_devices([], 'GPU')

preprocess_data = __import__('0-transfer').preprocess_data

# Charger les données CIFAR-10
(_, (X, Y)) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)

# Charger le modèle entraîné
model = K.models.load_model('cifar10.h5')

# Évaluer le modèle
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
