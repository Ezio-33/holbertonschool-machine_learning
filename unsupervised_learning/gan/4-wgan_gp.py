#!/usr/bin/env python3
"""
Wasserstein GAN avec pénalité de gradient pré-entraîné
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """
    Cette classe représente un Wasserstein GAN (WGAN)
        avec pénalité de gradient.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """
        Initialise le modèle WGAN-GP avec un générateur, un discriminateur,
        un générateur de vecteurs latents, des exemples réels et d'autres
        paramètres.
        """
        # Exécute d'abord __init__ de keras.Model.
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        # Valeur standard, mais peut être modifiée si nécessaire
        self.beta_1 = .3
        # Valeur standard, mais peut être modifiée si nécessaire
        self.beta_2 = .9

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # Définit la perte et l'optimiseur du générateur :
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss)

        # Définit la perte et l'optimiseur du discriminateur :
        self.discriminator.loss = lambda x, y: \
            tf.reduce_mean(y) - tf.reduce_mean(x)
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss)

    # Générateur d'échantillons réels de taille batch_size
    def get_fake_sample(self, size=None, training=False):
        """
        Génère un lot d'échantillons faux en utilisant le générateur.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # Générateur d'échantillons faux de taille batch_size
    def get_real_sample(self, size=None):
        """
        Récupère un lot d'échantillons réels à partir du jeu de données.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # Générateur d'échantillons interpolés de taille batch_size
    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Génère des échantillons interpolés entre
                des échantillons réels et faux.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    # Calcul de la pénalité de gradient
    def gradient_penalty(self, interpolated_sample):
        """
        Calcule la pénalité de gradient pour les échantillons interpolés.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    # Surcharge de train_step()
    def train_step(self, useless_argument):
        """
        Entraîne le discriminateur et le générateur du GAN et retourne leurs
        pertes respectives et la pénalité de gradient.
        """
        for _ in range(self.disc_iter):
            # Bande surveillant les poids du discriminateur
            with tf.GradientTape() as disc_tape:
                # Obtenir un lot d'échantillons réels
                real_samples = self.get_real_sample()
                # Obtenir un lot d'échantillons faux
                fake_samples = self.get_fake_sample(training=True)

                interpolated_sample = self.get_interpolated_sample(
                    real_samples, fake_samples)
                # Calculer la perte du discriminateur sur les échantillons
                # réels et faux
                real_preds = self.discriminator(real_samples, training=True)
                fake_preds = self.discriminator(fake_samples, training=True)
                discr_loss = self.discriminator.loss(real_preds, fake_preds)

                gp = self.gradient_penalty(interpolated_sample)
                new_discr_loss = discr_loss + self.lambda_gp * gp
            # Calculer les gradients pour le discriminateur
            discr_gradients = disc_tape.gradient(
                new_discr_loss, self.discriminator.trainable_variables)

            # Appliquer les gradients pour mettre à jour les poids du
            # discriminateur
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradients, self.discriminator.trainable_variables))

        # Bande surveillant les poids du générateur
        with tf.GradientTape() as gen_tape:
            # Obtenir un lot d'échantillons faux
            fake_samples = self.get_fake_sample(training=True)

            # Calculer la perte du générateur (à quel point le discriminateur
            # est trompé)
            fake_preds = self.discriminator(fake_samples, training=False)
            gen_loss = self.generator.loss(fake_preds)

        # Calculer les gradients pour le générateur
        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)

        # Appliquer les gradients pour mettre à jour les poids du générateur
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))

        # Retourner les pertes pour le discriminateur et le générateur
        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}

    def replace_weights(self, gen_h5, disc_h5):
        """
        Remplace les poids du générateur et du discriminateur par ceux stockés
        dans les fichiers locaux correspondants.
        """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)
