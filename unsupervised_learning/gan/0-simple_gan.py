#!/usr/bin/env python3
"""
Module Simple_GAN - Implémentation corrigée avec gestion des métriques
"""
import tensorflow as tf
from tensorflow import keras

class Simple_GAN(keras.Model):
    def __init__(self, generator, discriminator, latent_generator, real_examples, 
                 batch_size=200, disc_iter=2, learning_rate=0.005):
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

        # Déclaration des métriques
        self.disc_loss_tracker = keras.metrics.Mean(name="discr_loss")
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")

    def get_fake_sample(self, size=None, training=False):
        size = size or self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        size = size or self.batch_size
        indices = tf.range(tf.shape(self.real_examples)[0])
        return tf.gather(self.real_examples, tf.random.shuffle(indices)[:size])

    def train_step(self, data):
        # Réinitialisation des métriques
        self.disc_loss_tracker.reset_state()
        self.gen_loss_tracker.reset_state()

        # Entraînement du discriminateur
        for _ in range(self.disc_iter):
            with tf.GradientTape() as d_tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample()
                
                pred_real = self.discriminator(real_samples, training=True)
                pred_fake = self.discriminator(fake_samples, training=True)
                
                # Calcul de la perte Wasserstein
                d_loss = tf.reduce_mean(pred_fake) - tf.reduce_mean(pred_real)
            
            # Application des gradients
            d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
            self.disc_loss_tracker.update_state(d_loss)

        # Entraînement du générateur
        with tf.GradientTape() as g_tape:
            generated_samples = self.get_fake_sample()
            pred_fake = self.discriminator(generated_samples, training=False)
            
            # Calcul de la perte du générateur
            g_loss = -tf.reduce_mean(pred_fake)
        
        # Mise à jour des poids
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        self.gen_loss_tracker.update_state(g_loss)

        return {
            "discr_loss": self.disc_loss_tracker.result(),
            "gen_loss": self.gen_loss_tracker.result()
        }

    @property
    def metrics(self):
        return [self.disc_loss_tracker, self.gen_loss_tracker]
