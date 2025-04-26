#!/usr/bin/env python3
"""Variational Autoencoder (VAE) avec Keras"""

import tensorflow.keras as K

def sampling(args):
    """
    Reparamétrisation pour l'échantillonnage latent
    """
    z_mean, z_log_var = args
    batch = K.backend.shape(z_mean)[0]
    dim = K.backend.int_shape(z_mean)[1]
    epsilon = K.backend.random_normal(shape=(batch, dim))
    return z_mean + K.backend.exp(0.5 * z_log_var) * epsilon

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Construit un autoencodeur variationnel
    """
    # ========== Encodeur ==========
    inputs = K.Input(shape=(input_dims,), name='input')
    x = inputs
    
    # Construction des couches cachées
    for units in hidden_layers:
        x = K.layers.Dense(units, activation='relu')(x)
    
    # Couches de distribution latente
    z_mean = K.layers.Dense(latent_dims, name='z_mean')(x)
    z_log_var = K.layers.Dense(latent_dims, name='z_log_var')(x)
    
    # Échantillonnage avec reparamétrisation
    z = K.layers.Lambda(
        sampling, 
        output_shape=(latent_dims,), 
        name='z'
    )([z_mean, z_log_var])
    
    encoder = K.Model(
        inputs, 
        [z_mean, z_log_var, z], 
        name='encoder'
    )

    # ========== Décodeur ==========
    latent_inputs = K.Input(shape=(latent_dims,), name='z_sampling')
    x = latent_inputs
    
    # Construction inverse des couches
    for units in reversed(hidden_layers):
        x = K.layers.Dense(units, activation='relu')(x)
    
    # Reconstruction finale
    outputs = K.layers.Dense(
        input_dims, 
        activation='sigmoid', 
        name='decoder_output'
    )(x)
    
    decoder = K.Model(
        latent_inputs, 
        outputs, 
        name='decoder'
    )

    # ========== Modèle VAE complet ==========
    outputs = decoder(encoder(inputs)[2])
    vae = K.Model(inputs, outputs, name='vae')
    
    # Calcul de la perte KL
    kl_loss = -0.5 * K.backend.sum(
        1 + z_log_var - K.backend.square(z_mean) - K.backend.exp(z_log_var),
        axis=-1
    )
    vae.add_loss(K.backend.mean(kl_loss))
    
    # Compilation avec perte de reconstruction + KL
    vae.compile(
        optimizer='adam', 
        loss='binary_crossentropy'
    )
    
    return encoder, decoder, vae
