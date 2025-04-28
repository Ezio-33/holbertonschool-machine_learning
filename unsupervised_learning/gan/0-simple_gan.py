#!/usr/bin/env python3
"""
Module Simple_GAN - Implémentation corrigée avec gestion des métriques

Ce module contient la classe Simple_GAN qui implémente un réseau de neurones
génératif antagoniste (GAN) simple. Il inclut des fonctionnalités pour
l'entraînement du générateur et du discriminateur, ainsi que des méthodes
pour obtenir des échantillons réels et faux.
"""

import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """
    Classe Simple_GAN - Implémentation d'un GAN simple

    Cette classe hérite de keras.Model et implémente un réseau de neurones
    génératif antagoniste (GAN) simple. Elle inclut des méthodes pour
    l'entraînement du générateur et du discriminateur, ainsi que des méthodes
    pour obtenir des échantillons réels et faux.
    """

    def __init__(
            self,
            generator,
            discriminator,
            latent_generator,
            real_examples,
            batch_size=200,
            disc_iter=2,
            learning_rate=0.005):
        """
        Initialise une instance de Simple_GAN.

        Args:
            generator (keras.Model): Le modèle générateur.
            discriminator (keras.Model): Le modèle discriminateur.
            latent_generator (callable): Une fonction pour générer des
            vecteurs latents.
            real_examples (tf.Tensor): Les exemples réels pour l'entraînement.
            batch_size (int, optional): La taille du batch. Par défaut à 200.
            disc_iter (int, optional): Le nombre d'itérations pour
            l'entraînement du discriminateur. Par défaut à 2.
            learning_rate (float, optional): Le taux d'apprentissage.
            Par défaut à 0.005.
        """
        super().__init__()

        # Initialisation des composants
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        # Configuration des optimiseurs
        self.generator_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.5,
            beta_2=0.9
        )
        self.discriminator_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.5,
            beta_2=0.9
        )

        # Définition des fonctions de perte originales
        self.generator.loss = lambda x: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
        )
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
            tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape))
        )

        # Compilation des modèles
        self.generator.compile(
            optimizer=self.generator_optimizer,
            loss=self.generator.loss)
        self.discriminator.compile(
            optimizer=self.discriminator_optimizer,
            loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """
        Génère des échantillons faux.

        Args:
            size (int, optional): La taille des échantillons.
            Par défaut à None.
            training (bool, optional):
            Indique si le modèle est en mode entraînement.
            Par défaut à False.

        Returns:
            tf.Tensor: Les échantillons faux générés.
        """
        size = size or self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Génère des échantillons réels.

        Args:
            size (int, optional): La taille des échantillons.
            Par défaut à None.

        Returns:
            tf.Tensor: Les échantillons réels.
        """
        size = size or self.batch_size
        indices = tf.range(tf.shape(self.real_examples)[0])
        return tf.gather(self.real_examples, tf.random.shuffle(indices)[:size])

    def train_step(self, data):
        """
        Effectue une étape d'entraînement.

        Args:
            data: Les données d'entraînement.

        Returns:
            dict: Un dictionnaire contenant les pertes
            du discriminateur et du générateur.
        """
        # Entraînement du discriminateur
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample()

                pred_real = self.discriminator(real_samples, training=True)
                pred_fake = self.discriminator(fake_samples, training=True)

                disc_loss = self.discriminator.loss(pred_real, pred_fake)

            disc_grads = disc_tape.gradient(
                disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(
                zip(disc_grads, self.discriminator.trainable_variables)
            )

        # Entraînement du générateur
        with tf.GradientTape() as gen_tape:
            generated_samples = self.get_fake_sample()
            pred_fake = self.discriminator(generated_samples, training=False)

            gen_loss = self.generator.loss(pred_fake)

        gen_grads = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )

        return {"discr_loss": disc_loss, "gen_loss": gen_loss}
