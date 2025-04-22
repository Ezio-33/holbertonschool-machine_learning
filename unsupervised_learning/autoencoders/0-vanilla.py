#!/usr/bin/env python3
"""
Module pour implémenter un autoencodeur de base avec Keras
"""
import tensorflow.keras as keras

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Construit un autoencodeur simple
    
    Args:
        input_dims: Dimensions de l'entrée (ex: 784 pour MNIST)
        hidden_layers: Liste des neurones par couche cachée (ex: [128, 64])
        latent_dims: Dimension de l'espace latent (ex: 32)
        
    Returns:
        Tuple (encodeur, décodeur, autoencodeur complet)
    """
    # Construction de l'encodeur
    entree_encodeur = keras.layers.Input(shape=(input_dims,))
    x = entree_encodeur
    
    # Ajout des couches cachées
    for neurones in hidden_layers:
        x = keras.layers.Dense(neurones, activation='relu')(x)
    
    # Couche latente (goulot d'étranglement)
    sortie_latente = keras.layers.Dense(latent_dims, activation='relu')(x)
    
    # Modèle encodeur final
    encodeur = keras.models.Model(
        inputs=entree_encodeur, 
        outputs=sortie_latente,
        name="encodeur"
    )

    # Construction du décodeur
    entree_decodeur = keras.layers.Input(shape=(latent_dims,))
    x = entree_decodeur
    
    # Ajout des couches en ordre inverse
    for neurones in reversed(hidden_layers):
        x = keras.layers.Dense(neurones, activation='relu')(x)
    
    # Couche de reconstruction finale
    sortie_decodeur = keras.layers.Dense(
        input_dims, 
        activation='sigmoid'  # Sortie entre 0 et 1 comme les images normalisées
    )(x)
    
    # Modèle décodeur final
    decodeur = keras.models.Model(
        inputs=entree_decodeur, 
        outputs=sortie_decodeur,
        name="decodeur"
    )

    # Assemblage de l'autoencodeur complet
    entree_autoencodeur = entree_encodeur
    sortie_autoencodeur = decodeur(encodeur(entree_autoencodeur))
    
    autoencodeur_complet = keras.models.Model(
        inputs=entree_autoencodeur,
        outputs=sortie_autoencodeur,
        name="autoencodeur_complet"
    )
    
    # Configuration de l'apprentissage
    autoencodeur_complet.compile(
        optimizer='adam',
        loss='binary_crossentropy'  # Mesure la différence pixel à pixel
    )
    
    return encodeur, decodeur, autoencodeur_complet
