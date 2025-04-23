#!/usr/bin/env python3
"""
Module pour implémenter un autoencodeur parcimonieux avec Keras
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Construit un autoencodeur avec régularisation L1

    Args:
        input_dims: Dimensions de l'entrée (ex: 784)
        hidden_layers: Liste des neurones par couche (ex: [128, 64])
        latent_dims: Dimension de l'espace latent (ex: 32)
        lambtha: Coefficient de régularisation L1

    Returns:
        Tuple (encodeur, décodeur, autoencodeur)
    """
    # Partie Encodeur
    entree_encodeur = keras.layers.Input(shape=(input_dims,))
    x = entree_encodeur

    # Construction des couches cachées
    for neurones in hidden_layers:
        x = keras.layers.Dense(neurones, activation='relu')(x)

    # Couche latente avec régularisation L1
    sortie_latente = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(
            lambtha)  # Régularisation ici
    )(x)

    encodeur = keras.models.Model(
        inputs=entree_encodeur,
        outputs=sortie_latente,
        name="encodeur"
    )

    # Partie Décodeur (identique à la tâche 0)
    entree_decodeur = keras.layers.Input(shape=(latent_dims,))
    x = entree_decodeur

    for neurones in reversed(hidden_layers):
        x = keras.layers.Dense(neurones, activation='relu')(x)

    sortie_decodeur = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decodeur = keras.models.Model(
        inputs=entree_decodeur,
        outputs=sortie_decodeur,
        name="decodeur"
    )

    # Assemblage final
    entree_autoencodeur = entree_encodeur
    sortie_autoencodeur = decodeur(encodeur(entree_autoencodeur))

    autoencodeur_complet = keras.models.Model(
        inputs=entree_autoencodeur,
        outputs=sortie_autoencodeur,
        name="autoencodeur_parcimonieux"
    )

    # Compilation (identique à la tâche 0)
    autoencodeur_complet.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encodeur, decodeur, autoencodeur_complet
