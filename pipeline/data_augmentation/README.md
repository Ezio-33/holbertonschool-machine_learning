# Data Augmentation

Ce dossier contient des fonctions simples de data augmentation utilisant TensorFlow 2.15.

Prérequis:

- Python 3.9
- TensorFlow 2.15
- Numpy 1.25.2

Installation optionnelle (pour les scripts d'exemple):

- tensorflow-datasets==4.9.2
- matplotlib

Fichiers fournis:

- 0-flip.py: flip_image(image) — miroir horizontal
- 1-crop.py: crop_image(image, size) — recadrage aléatoire
- 2-rotate.py: rotate_image(image) — rotation 90° anti-horaire
- 3-contrast.py: change_contrast(image, lower, upper) — contraste aléatoire
- 4-brightness.py: change_brightness(image, max_delta) — luminosité aléatoire
- 5-hue.py: change_hue(image, delta) — modification de la teinte

Notes:

- Chaque fichier commence par le shebang et n'importe que `import tensorflow as tf`.
- Chaque module et fonction est documenté et exécutable.
- Les fonctions prennent des tenseurs 3D HxWxC en entrée et renvoient un tenseur transformé de même type.
