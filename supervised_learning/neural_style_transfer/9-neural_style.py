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
        self.generate_features()

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

    def generate_features(self):
        """
        Extrait les caractéristiques utilisées pour
        calculer le coût de style neural.
        Définit les attributs d'instance publics :
            - gram_style_features - une liste de matrices de Gram calculées à
            partir des sorties des couches de style de l'image de style.
            - content_feature - la sortie de la couche de contenu de
            l'image de contenu.
        """
        # Assure que les images sont correctement prétraitées
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

# Obtient les sorties du modèle avec les images prétraitées comme entrée
        style_outputs = self.model(preprocessed_style)[:-1]

# Définit content_feature, aucun traitement supplémentaire n'est requis
        self.content_feature = self.model(preprocessed_content)[-1]

# Calcule et définit les matrice de Gram pour les sorties des couches de style
        self.gram_style_features = [self.gram_matrix(
            output) for output in style_outputs]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calcule le coût de style pour une couche spécifique

        Args:
            style_output: tf.Tensor de forme (1, h, w, c) contenant la sortie
                          de style de la couche pour l'image générée
            gram_target: tf.Tensor de forme (1, c, c) la matrice de Gram de
                        la sortie de style cible pour cette couche

        Returns:
            Le coût de style de la couche
        """
        # Validation des entrées
        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        # Extraire c à partir de style_output, pas de gram_target
        c = style_output.shape[-1]

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or gram_target.shape != (1, c, c)):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]")

        # Calcul de la matrice de Gram générée
        gram_generated = self.gram_matrix(style_output)

        # S'assurer que les types correspondent
        gram_generated = tf.cast(gram_generated, dtype=gram_target.dtype)

        # Retourne la différence quadratique moyenne
        return tf.reduce_mean(tf.square(gram_generated - gram_target))

    def style_cost(self, style_outputs):
        """
        Calcule le coût de style total en combinant toutes les couches
        """
        len_s = len(self.style_layers)
        # Validation des entrées
        if not isinstance(style_outputs, list) or len(style_outputs) != len_s:
            raise TypeError(
                f"style_outputs must be a list with a length of {len_s}")

        total_cost = 0.0
        weight = 1.0 / len_s

        for style_out, gram in zip(style_outputs, self.gram_style_features):
            layer_cost = self.layer_style_cost(style_out, gram)
            total_cost += weight * layer_cost

        return total_cost

    def content_cost(self, content_output):
        """
        Calcule le coût de contenu avec la bonne normalisation
        """
        scf = self.content_feature.shape
        # Validation de la forme
        if (not isinstance(content_output, (tf.Tensor, tf.Variable))
                or content_output.shape != scf):
            raise TypeError(f"content_output must be a tensor of shape {scf}")

        # Calcul de la différence quadratique moyenne
        return tf.reduce_mean(
            tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """
        Calcule le coût total combinant contenu et style

        Args:
            generated_image: Tenseur TF de forme (1, H, W, 3)

        Returns:
            tuple: (Coût_total, Coût_contenu, Coût_style)
        """
        # Vérification de la forme de l'image générée
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or s != generated_image.shape):
            raise TypeError(f"generated_image must be a tensor of shape {s}")

        # Prétraitement pour VGG19
        generated_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0)

        # Extraction des caractéristiques
        outputs = self.model(generated_preprocessed)
        style_outputs = outputs[:-1]  # Sorties des couches de style
        content_output = outputs[-1]  # Sortie de la couche de contenu

        # Calcul des coûts
        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J_total = self.alpha * J_content + self.beta * J_style

        return J_total, J_content, J_style

    def compute_grads(self, generated_image):
        """
        Calcule les gradients du coût total par rapport à l'image générée

        Args:
            generated_image: tf.Tensor de forme (1, H, W, 3)

        Returns:
            tuple: (gradients, J_total, J_content, J_style)
        """
        # Vérification de la forme
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != s):
            raise TypeError(f"generated_image must be a tensor of shape {s}")

        # Calcul des gradients avec GradientTape
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01, beta1=0.9,
                       beta2=0.99):
        """
        Génère l'image de transfert de style

        Args:
            iterations: Nombre d'itérations d'optimisation
            step: Intervalle d'affichage des coûts
            lr: Taux d'apprentissage
            beta1: Paramètre beta1 d'Adam
            beta2: Paramètre beta2 d'Adam

        Returns:
            best_image: Image optimisée
            best_cost: Meilleur coût total
        """
        # Validation des entrées
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "step must be positive and less than iterations")
        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if not (0 <= beta1 <= 1):
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if not (0 <= beta2 <= 1):
            raise ValueError("beta2 must be in the range [0, 1]")

        # Initialisation avec l'image de contenu
        generated_image = tf.Variable(self.content_image, dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2
        )

        best_cost = float('inf')
        best_image = None

        # Boucle d'optimisation
        for i in range(iterations):
            with tf.GradientTape() as tape:
                J_total, J_content, J_style = self.total_cost(generated_image)

            # Mise à jour de l'image
            gradients = tape.gradient(J_total, generated_image)
            optimizer.apply_gradients([(gradients, generated_image)])

            # Clipping des valeurs entre 0 et 1
            generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))

            # Sauvegarde du meilleur résultat
            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.numpy()

            # Affichage périodique
            if step and (i + 1) % step == 0:
                print(f"Iteration {i+1}: Coût={J_total}, "
                      f"Contenu={J_content}, Style={J_style}")

        return best_image[0], best_cost
