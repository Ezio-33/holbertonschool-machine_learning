#!/usr/bin/env python3
"""Implémentation de ResNet-50 pour le projet Deep Convolutional Architectures

Ce module construit le modèle ResNet-50 tel que décrit dans
'Deep Residual Learning for Image Recognition' (2015).

Chaque couche de convolution est suivie d’une normalisation par lots (Batch Normalization)
et d’une activation ReLU. L’initialisation des poids utilise la méthode He Normal avec un seed fixe,
ce qui garantit la reproductibilité des résultats.
"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Construit le modèle ResNet-50.

    Returns:
        Un modèle Keras représentant ResNet-50.
    """
    activation = 'relu'
    # Initialisation He Normal avec un seed fixe pour être reproductible
    kernel_init = K.initializers.HeNormal(seed=0)

    # Entrée du modèle : images de taille 224x224 avec 3 canaux de couleur
    # (RGB)
    X = K.Input(shape=(224, 224, 3))

    # Couche initiale : convolution 7x7 avec 64 filtres et stride de 2
    layer1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                             padding='same', kernel_initializer=kernel_init)(X)
    # Normalisation par lots sur l'axe des canaux (channels-last)
    batchNorm_l1 = K.layers.BatchNormalization(axis=3)(layer1)
    # Activation ReLU
    activation1 = K.layers.Activation(activation)(batchNorm_l1)
    # Max pooling : réduit la dimension spatiale en conservant les
    # informations essentielles
    layer_pool1 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                     padding='same')(activation1)

    # --- Étage 1 (3 blocs) ---
    # Premier bloc : Projection block pour adapter les dimensions (sans
    # réduction de résolution, s=1)
    layer2 = projection_block(layer_pool1, [64, 64, 256], s=1)
    # Ensuite, deux Identity blocks qui conservent la même dimension
    layer3 = identity_block(layer2, [64, 64, 256])
    layer4 = identity_block(layer3, [64, 64, 256])

    # --- Étage 2 (4 blocs) ---
    # Projection block pour augmenter le nombre de filtres et réduire la
    # taille spatiale
    layer5 = projection_block(layer4, [128, 128, 512])
    # Suivi de trois Identity blocks
    layer6 = identity_block(layer5, [128, 128, 512])
    layer7 = identity_block(layer6, [128, 128, 512])
    layer8 = identity_block(layer7, [128, 128, 512])

    # --- Étage 3 (6 blocs) ---
    layer9 = projection_block(layer8, [256, 256, 1024])
    layer10 = identity_block(layer9, [256, 256, 1024])
    layer11 = identity_block(layer10, [256, 256, 1024])
    layer12 = identity_block(layer11, [256, 256, 1024])
    layer13 = identity_block(layer12, [256, 256, 1024])
    layer14 = identity_block(layer13, [256, 256, 1024])

    # --- Étage 4 (3 blocs) ---
    layer15 = projection_block(layer14, [512, 512, 2048])
    layer16 = identity_block(layer15, [512, 512, 2048])
    layer17 = identity_block(layer16, [512, 512, 2048])

    # Couches finales : pooling global qui résume chaque feature map par une
    # moyenne
    average_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7), padding='same')(layer17)

    # Couche dense finale avec softmax pour la classification en 1000 classes
    # (ImageNet)
    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=kernel_init)(average_pool)

    # Création du modèle en reliant l'entrée et la sortie
    model = K.models.Model(inputs=X, outputs=Y)

    return model
