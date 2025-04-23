#!/usr/bin/env python3
"""
Module pour implémenter un autoencodeur convolutif avec Keras
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Construit un autoencodeur convolutif

    Args:
        input_dims: Tuple (hauteur, largeur, canaux) de l'entrée
        filters: Liste du nombre de filtres par couche convolutive
        latent_dims: Dimensions de l'espace latent

    Returns:
        Tuple (encodeur, décodeur, autoencodeur complet)
    """
    # Partie Encodeur
    entree_encodeur = keras.layers.Input(shape=input_dims)
    x = entree_encodeur

    # Construction des couches convolutives
    for n_filtres in filters:
        x = keras.layers.Conv2D(
            n_filtres,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        )(x)
        x = keras.layers.MaxPooling2D(
            (2, 2),
            padding='same'
        )(x)

    encodeur = keras.models.Model(
        inputs=entree_encodeur,
        outputs=x,
        name="encodeur_convolutif"
    )

    # Partie Décodeur
    entree_decodeur = keras.layers.Input(shape=latent_dims)
    x = entree_decodeur

    # Reconstruction inverse avec gestion des dimensions
    for n_filtres in reversed(filters[1:]):  # On ignore le premier filtre
        x = keras.layers.Conv2D(
            n_filtres,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    # Couche spéciale pour ajuster les dimensions
    x = keras.layers.Conv2D(
        filters[0],
        kernel_size=(3, 3),
        activation='relu',
        padding='valid'  # Padding différent ici
    )(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    # Dernière couche de reconstruction
    sortie_decodeur = keras.layers.Conv2D(
        input_dims[2],  # Même nombre de canaux que l'entrée
        kernel_size=(3, 3),
        activation='sigmoid',
        padding='same'
    )(x)

    decodeur = keras.models.Model(
        inputs=entree_decodeur,
        outputs=sortie_decodeur,
        name="decodeur_convolutif"
    )

    # Assemblage final
    autoencodeur_complet = keras.models.Model(
        inputs=entree_encodeur,
        outputs=decodeur(encodeur(entree_encodeur)),
        name="autoencodeur_convolutif"
    )

    # Configuration de l'apprentissage
    autoencodeur_complet.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encodeur, decodeur, autoencodeur_complet
