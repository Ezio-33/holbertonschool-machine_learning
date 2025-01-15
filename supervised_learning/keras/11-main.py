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
model = __import__('9-model')
config = __import__('11-config')

if __name__ == '__main__':
    network = model.load_model('network1.keras')
    config.save_config(network, 'config1.json')
    del network

    network2 = config.load_config('config1.json')
    network2.summary()
    print(network2.get_weights())
