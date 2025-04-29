#!/usr/bin/env python3
"""
Module WGAN_clip - Implémentation de Wasserstein GAN avec weight clipping
"""
import tensorflow as tf
from tensorflow import keras


class WGAN_clip(keras.Model):
    """Classe WGAN avec contrainte de clipping des poids"""

    def __init__(
            self,
            generator,
            discriminator,
            latent_generator,
            real_examples,
            batch_size=200,
            disc_iter=2,
            learning_rate=0.005):
        super().__init__()

        # Initialisation des composants
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        # Configuration des optimiseurs
        self.g_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.5,
            beta_2=0.9
        )
        self.d_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.5,
            beta_2=0.9
        )

        # Définition des fonctions de perte Wasserstein
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.discriminator.loss = lambda real_out, fake_out: tf.reduce_mean(
            fake_out) - tf.reduce_mean(real_out)

    def get_fake_sample(self, size=None, training=False):
        """Génère un batch de données synthétiques"""
        size = size or self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Sélectionne un batch de données réelles"""
        size = size or self.batch_size
        indices = tf.range(tf.shape(self.real_examples)[0])
        return tf.gather(self.real_examples, tf.random.shuffle(indices)[:size])

    def train_step(self, data):
        """Une étape complète d'entraînement"""

        # Entraînement du discriminateur
        for _ in range(self.disc_iter):
            with tf.GradientTape() as d_tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample()

                real_out = self.discriminator(real_samples, training=True)
                fake_out = self.discriminator(fake_samples, training=True)

                d_loss = self.discriminator.loss(real_out, fake_out)

            # Calcul et application des gradients avec clipping
            d_gradients = d_tape.gradient(
                d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradients, self.discriminator.trainable_variables))

            # Clipping des poids après chaque mise à jour
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        # Entraînement du générateur
        with tf.GradientTape() as g_tape:
            generated_samples = self.get_fake_sample()
            fake_out = self.discriminator(generated_samples, training=False)

            g_loss = self.generator.loss(fake_out)

        # Mise à jour des poids du générateur
        g_gradients = g_tape.gradient(
            g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}
