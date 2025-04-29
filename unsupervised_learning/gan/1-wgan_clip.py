#!/usr/bin/env python3
"""
Module WGAN_clip - Implémentation WGAN avec weight clipping
"""
import tensorflow as tf
from tensorflow import keras


class WGAN_clip(keras.Model):
    """
    Classe WGAN avec contrainte de clipping des poids

    Args:
        generator (keras.Model): Modèle générateur
        discriminator (keras.Model): Modèle discriminateur
        latent_generator (callable): Générateur de vecteurs latents
        real_examples (tf.Tensor): Jeu de données réel
        batch_size (int): Taille du batch
        disc_iter (int): Nombre d'itérations du discriminateur
        learning_rate (float): Taux d'apprentissage
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
        super().__init__()

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

    def get_fake_sample(self, size=None, training=False):
        """Génère un échantillon de données synthétiques"""
        size = size or self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Sélectionne un échantillon de données réelles"""
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

                # Calcul des prédictions
                pred_real = self.discriminator(real_samples, training=True)
                pred_fake = self.discriminator(fake_samples, training=True)

                # Calcul de la perte Wasserstein
                d_loss = tf.reduce_mean(pred_fake) - tf.reduce_mean(pred_real)

            # Application des gradients
            d_grads = d_tape.gradient(
                d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_grads, self.discriminator.trainable_variables))

            # Clipping des poids
            for w in self.discriminator.trainable_variables:
                w.assign(tf.clip_by_value(w, -1.0, 1.0))

        # Entraînement du générateur
        with tf.GradientTape() as g_tape:
            generated_samples = self.get_fake_sample()
            pred_fake = self.discriminator(generated_samples, training=False)

            # Calcul de la perte du générateur
            g_loss = -tf.reduce_mean(pred_fake)

        # Mise à jour des poids
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}
