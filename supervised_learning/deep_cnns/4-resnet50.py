#!/usr/bin/env python3
"""
Implémentation de ResNet-50 pour le projet Deep Convolutional Architectures
Ce module construit le modèle ResNet-50 tel que décrit dans
'Deep Residual Learning for Image Recognition' (2015).
"""

from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Function that builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)

    Returns:
        The keras model
    """
    # Définition de la fonction d'activation pour toutes les couches
    # convolutives.
    activation = 'relu'
    # Initialisation He normal avec seed=0 pour la reproductibilité (demandé
    # par la consigne).
    kernel_init = K.initializers.he_normal(seed=0)

    # Couche d'entrée, on suppose des images de taille 224x224 avec 3 canaux
    # (RGB).
    X = K.Input(shape=(224, 224, 3))

    # --- Couche initiale ---
    # Convolution 7x7 avec 64 filtres, stride 2 et padding 'same' :
    # Cela diminue la taille spatiale tout en extrayant des caractéristiques
    # de bas niveau.
    layer1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                             padding='same', kernel_initializer=kernel_init)(X)
    # Batch Normalization sur l’axe des canaux pour stabiliser l'apprentissage.
    batchNorm_l1 = K.layers.BatchNormalization(axis=3)(layer1)
    # Activation ReLU pour introduire la non-linéarité.
    activation1 = K.layers.Activation(activation)(batchNorm_l1)
    # Max Pooling 3x3 avec stride 2 et padding 'same' pour réduire
    # spatialement les dimensions.
    layer_pool1 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                     padding='same')(activation1)

    # --- Étape 1 (Étage 1) ---
    # Premier bloc : Projection Block qui ajuste les dimensions (ici, on ne
    # change pas la résolution car stride=1).
    layer2 = projection_block(layer_pool1, [64, 64, 256], 1)
    # Deux blocs d'identité (Identity Blocks) qui conservent la dimension.
    layer3 = identity_block(layer2, [64, 64, 256])
    layer4 = identity_block(layer3, [64, 64, 256])

    # --- Étape 2 (Étage 2) ---
    # Projection Block pour ajuster les dimensions et augmenter le nombre de
    # canaux à 512.
    layer5 = projection_block(layer4, [128, 128, 512])
    # Trois blocs d'identité successifs.
    layer6 = identity_block(layer5, [128, 128, 512])
    layer7 = identity_block(layer6, [128, 128, 512])
    layer8 = identity_block(layer7, [128, 128, 512])

    # --- Étape 3 (Étage 3) ---
    # Projection Block pour passer aux dimensions de 1024 canaux.
    layer9 = projection_block(layer8, [256, 256, 1024])
    # Cinq blocs d'identité successifs pour approfondir l'apprentissage des
    # caractéristiques.
    layer10 = identity_block(layer9, [256, 256, 1024])
    layer11 = identity_block(layer10, [256, 256, 1024])
    layer12 = identity_block(layer11, [256, 256, 1024])
    layer13 = identity_block(layer12, [256, 256, 1024])
    layer14 = identity_block(layer13, [256, 256, 1024])

    # --- Étape 4 (Étage 4) ---
    # Projection Block final pour passer aux dimensions de 2048 canaux.
    layer15 = projection_block(layer14, [512, 512, 2048])
    # Deux blocs d'identité finaux sur la dimension 2048.
    layer16 = identity_block(layer15, [512, 512, 2048])
    layer17 = identity_block(layer16, [512, 512, 2048])

    # --- Couches finales ---
    # Global Average Pooling pour réduire les feature maps à une seule valeur
    # par canal.
    average_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7), padding='same')(layer17)
    # Couche Dense finale avec softmax pour la classification sur 1000 classes.
    Y = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=kernel_init)(average_pool)
    # Construction du modèle en reliant l'entrée à la sortie.
    model = K.models.Model(inputs=X, outputs=Y)

    return model
