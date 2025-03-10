#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
import tensorflow as tf
import numpy as np
import random
import os
SEED = 8

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Imports
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model
model = __import__('9-model')
weights = __import__('10-weights')

if __name__ == '__main__':

    network = model.load_model('network2.keras')
    weights.save_weights(network, 'weights2.keras')
    del network

    network2 = model.load_model('network1.keras')
    print(network2.get_weights())
    weights.load_weights(network2, 'weights2.keras')
    print(network2.get_weights())
