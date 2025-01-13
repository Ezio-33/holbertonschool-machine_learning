#!/usr/bin/env python3
"""
Module pour créer les placeholders d'un réseau de neurones
"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Crée les placeholders pour le réseau de neurones
    
    Args:
        nx: nombre de caractéristiques d'entrée
        classes: nombre de classes de classification
        
    Returns:
        x: placeholder pour les données d'entrée
        y: placeholder pour les étiquettes one-hot
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
