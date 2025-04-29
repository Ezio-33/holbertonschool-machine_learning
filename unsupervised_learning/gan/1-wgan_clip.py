#!/usr/bin/env python3
"""
Module WGAN_clip - Implémentation de Wasserstein GAN avec poids clipsés
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """
    Classe avec une nouvelle fonction de perte et un clipping
    Je ne sais pas quoi faire !
    """

    def __init__(self, generator, discriminator,
                 latent_generator, real_examples,
                 batch_size=200, disc_iter=2,
                 learning_rate=.005):
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # Définir la perte et l'optimiseur du générateur :
        self.generator.loss = lambda x: \
            - tf.math.reduce_mean(input_tensor=x)

        self.generator.optimizer = keras.optimizers.\
            Adam(learning_rate=self.learning_rate,
                 beta_1=self.beta_1,
                 beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # Définir la perte et l'optimiseur du discriminateur :
        self.discriminator.loss =\
            lambda x, y: - tf.math.reduce_mean(input_tensor=x)\
            + tf.math.reduce_mean(input_tensor=y)

        self.discriminator.optimizer =\
            keras.optimizers.Adam(learning_rate=self.
                                  learning_rate,
                                  beta_1=self.beta_1,
                                  beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer,
                                   loss=discriminator.loss)

    # Générateur d'échantillons réels de taille batch_size
    def get_fake_sample(self, size=None, training=False):
        """
        Obtenir un échantillon faux
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # Générateur d'échantillons faux de taille batch_size
    def get_real_sample(self, size=None):
        """
        Obtenir un échantillon réel
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        Utilisé pour entraîner le générateur et le discriminateur
        """
        # 1. Entraînement du discriminateur
        for _ in range(self.disc_iter):

            # Calculer la perte pour le discriminateur
            # dans une bande surveillant les poids du discriminateur
            with tf.GradientTape() as tape:

                # Obtenir un échantillon réel
                real = self.get_real_sample(size=None)

                # Obtenir un échantillon faux
                fake = self.get_fake_sample(size=None)

                # Calculer la perte discr_loss
                # du discriminateur sur les échantillons réels et faux
                pred_real = self.discriminator(real, training=True)
                pred_fake = self.discriminator(fake, training=True)
                discr_loss = self.discriminator.loss(pred_real, pred_fake)

            # Appliquer la descente de gradient une fois au discriminateur
            discr_grads = tape.gradient(discr_loss,
                                        self.discriminator.trainable_variables)
            self.discriminator.optimizer.\
                apply_gradients(zip(discr_grads,
                                    self.discriminator.trainable_variables))

            for w in self.discriminator.trainable_weights:
                w.assign(tf.clip_by_value(w, -1, 1))

        # 2. Entraînement du générateur
        with tf.GradientTape() as tape:
            # Obtenir un échantillon faux
            fake = self.get_fake_sample(size=None)

            # Calculer la perte gen_loss du générateur sur cet échantillon
            pred_fake = self.discriminator(fake, training=True)
            gen_loss = self.generator.loss(pred_fake)

        # Appliquer la descente de gradient au générateur
        gen_grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.\
            apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        # Retourner {"discr_loss": discr_loss, "gen_loss": gen_loss}
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
