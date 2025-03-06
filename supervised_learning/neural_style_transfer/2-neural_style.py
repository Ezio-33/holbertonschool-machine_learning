#!/usr/bin/env python3
"""
Transfert de Style Neural
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    La classe NST effectue des tâches pour le transfert de style neural.

    Attributs de Classe Publique :
    - style_layers: Une liste de couches à utiliser
    pour l'extraction de style,par défaut ['block1_conv1',
    'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'].
    - content_layer: La couche à utiliser pour l'extraction de contenu,
      par défaut 'block5_conv2'.
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initialise une instance de NST.

        Paramètres :
        - style_image (numpy.ndarray):
        L'image utilisée comme référence de style.
        - content_image (numpy.ndarray):
        L'image utilisée comme référence de contenu.
        - alpha (float): Le poids pour le coût de contenu. Par défaut 1e4.
        - beta (float): Le poids pour le coût de style. Par défaut 1.

        Lève :
        - TypeError: Si style_image n'est pas un numpy.ndarray avec
          une forme (h, w, 3), lève une erreur avec le message "style_image
          doit être un numpy.ndarray avec une forme (h, w, 3)".
        - TypeError: Si content_image n'est pas un numpy.ndarray avec
          une forme (h, w, 3), lève une erreur avec le message "content_image
          doit être un numpy.ndarray avec une forme (h, w, 3)".
        - TypeError: Si alpha n'est pas un nombre non négatif, lève une erreur
          avec le message "alpha doit être un nombre non négatif".
        - TypeError: Si beta n'est pas un nombre non négatif, lève une erreur
          avec le message "beta doit être un nombre non négatif".

        Attributs d'Instance :
        - style_image: L'image de style prétraitée.
        - content_image: L'image de contenu prétraitée.
        - alpha: Le poids pour le coût de contenu.
        - beta: Le poids pour le coût de style.
        """
        if (not isinstance(style_image, np.ndarray)
                or style_image.shape[-1] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (float, int)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (float, int)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Redimensionne une image de sorte que ses
        valeurs de pixels soient entre 0 et 1 et que son
        plus grand côté soit de 512 pixels.

        Paramètres :
        - image (numpy.ndarray): Un numpy.ndarray de forme (h, w, 3) contenant
          l'image à redimensionner.

        Lève :
        - TypeError: Si image n'est pas un numpy.ndarray
        avec une forme (h, w, 3), lève une erreur avec le
        message "image doit être un numpy.ndarray avec une forme (h, w, 3)".

        Retourne :
        - tf.Tensor: L'image redimensionnée sous
          forme de tf.Tensor avec une forme
          (1, h_new, w_new, 3), où max(h_new, w_new) == 512 et
          min(h_new, w_new) est redimensionné proportionnellement.
          L'image est redimensionnée en utilisant une interpolation bicubique,
          et ses valeurs de pixels sont redimensionnées de
          la plage [0, 255] à [0, 1].
        """
        if (not isinstance(image, np.ndarray) or image.shape[-1] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w = image.shape[:2]

        if w > h:
            new_w = 512
            new_h = int((h * 512) / w)
        else:
            new_h = 512
            new_w = int((w * 512) / h)

        # Redimensionne l'image (avec interpolation bicubique)
        image_resized = tf.image.resize(
            image, size=[new_h, new_w],
            method=tf.image.ResizeMethod.BICUBIC)

        # Normalise les valeurs de pixels dans la plage [0, 1]
        image_normalized = image_resized / 255

        # Limite les valeurs pour s'assurer qu'elles sont dans la plage [0, 1]
        image_clipped = tf.clip_by_value(image_normalized, 0, 1)

        # Ajoute une dimension de lot sur l'axe 0 et retourne
        return tf.expand_dims(image_clipped, axis=0)

    def load_model(self):
        """
        Charge le modèle VGG19 avec des couches AveragePooling2D au lieu
        de couches MaxPooling2D.
        """
        # Obtient VGG19 depuis l'API Keras
        modelVGG19 = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        modelVGG19.trainable = False

        # Couches sélectionnées
        selected_layers = self.style_layers + [self.content_layer]

        outputs = [modelVGG19.get_layer(name).output for name
                   in selected_layers]

        # Construit le modèle
        model = tf.keras.Model([modelVGG19.input], outputs)

        # Remplace les couches MaxPooling par des couches AveragePooling
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        tf.keras.models.save_model(model, 'vgg_base.h5')
        model_avg = tf.keras.models.load_model('vgg_base.h5',
                                               custom_objects=custom_objects)

        self.model = model_avg

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calcule la matrice de Gram d'un tenseur donné.

        Args :
        - input_layer: Un tf.Tensor ou tf.Variable de forme (1, h, w, c).

        Retourne :
        - Un tf.Tensor de forme (1, c, c) contenant la matrice de Gram de
          input_layer.
        """
        # Valide le rang et la taille du lot de input_layer
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4
                or input_layer.shape[0] != 1):
            raise TypeError("input_layer must be a tensor of rank 4")

        # Calcule la matrice de Gram : (batch, hauteur, largeur, canal)
        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

# Normalise par le nombre d'emplacements(h*w)puis retourne le tenseur de Gram
        input_shape = tf.shape(input_layer)
        nb_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return gram / nb_locations
