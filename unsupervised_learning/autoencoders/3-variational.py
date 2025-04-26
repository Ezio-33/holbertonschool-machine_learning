#!/usr/bin/env python3
"""3-variational.py"""
import tensorflow.keras as K


def sampling(args):
    """
    Reparamétrisation trick pour échantillonner z ~ N(mu, sigma^2)

    Args:
        args: tuple contenant (z_mean, z_log_var)
    Returns:
        z: vecteur latent échantillonné selon N(mu, sigma^2)
    """
    # Dépaquetage des paramètres de distribution
    z_mean, z_log_var = args

    # Génération d'un bruit aléatoire gaussien
    epsilon = K.backend.random_normal(shape=K.backend.shape(z_mean))

    # Calcul de l'écart-type à partir du log variance
    sigma = K.backend.exp(0.5 * z_log_var)

    # Formule de reparamétrisation: z = mu + sigma * epsilon
    return z_mean + sigma * epsilon


def vae_loss(inputs, outputs, z_mean, z_log_var, input_dims):
    """
    Calcul de la perte totale du VAE (reconstruction + KL divergence)

    Args:
        inputs: données d'entrée originales
        outputs: données reconstruites
        z_mean: moyenne de la distribution latente
        z_log_var: log variance de la distribution latente
        input_dims: dimensions des données d'entrée
    """
    # Perte de reconstruction (erreur de reconstruction pixel à pixel)
    reconstruction_loss = K.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dims  # Normalisation par la dimension

    # Calcul de la divergence KL entre N(mu, sigma) et N(0, 1)
    kl_loss = 1 + z_log_var - \
        K.backend.square(z_mean) - K.backend.exp(z_log_var)
    kl_loss = -0.5 * K.backend.sum(kl_loss, axis=-1)

    # Combinaison des deux pertes
    return K.backend.mean(reconstruction_loss + kl_loss)


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Construction complète du VAE"""
    # === Partie Encodeur ===
    # Couche d'entrée pour les données brutes
    inputs = K.Input(shape=(input_dims,))

    # Construction des couches cachées de l'encodeur
    x = inputs
    for nodes in hidden_layers:
        x = K.layers.Dense(nodes, activation='relu')(x)

    # Sorties de l'encodeur: mu et log_var
    z_mean = K.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = K.layers.Dense(latent_dims, activation=None)(x)

    # Échantillonnage avec reparamétrisation
    z = K.layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    # Création du modèle encodeur avec triple sortie
    encoder = K.Model(
        inputs,
        [z, z_mean, z_log_var],
        name="encoder"
    )

    # === Partie Décodeur ===
    # Entrée pour l'espace latent
    latent_inputs = K.Input(shape=(latent_dims,))

    # Construction inverse des couches
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = K.layers.Dense(nodes, activation='relu')(x)

    # Couche de reconstruction finale avec sigmoid
    decoded = K.layers.Dense(input_dims, activation='sigmoid')(x)

    # Création du modèle décodeur
    decoder = K.Model(latent_inputs, decoded, name="decoder")

    # === Assemblage du VAE ===
    # Chaînage encodeur -> décodeur
    # On prend uniquement l'échantillon z
    outputs = decoder(encoder(inputs)[0])

    # Création du modèle final
    auto = K.Model(inputs, outputs, name="vae")

    # Ajout de la perte personnalisée au modèle
    auto.add_loss(
        vae_loss(inputs, outputs, z_mean, z_log_var, input_dims)
    )

    # Compilation avec Adam (version objet pour meilleur contrôle)
    auto.compile(optimizer=K.optimizers.Adam(learning_rate=0.001))

    return encoder, decoder, auto
