#!/usr/bin/env python3
"""
Optimisation bayésienne d'un CNN sur MNIST avec GPyOpt
"""

import GPyOpt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import matplotlib.pyplot as plt

# Configuration de base
np.random.seed(42)
tf.random.set_seed(42)

def build_model(params):
    """Construit un CNN """
    lr, filters, dense_units, dropout, l2 = params[0]

    model = models.Sequential([
        layers.Conv2D(filters, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(l2)),
        layers.Dropout(dropout),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def objective_function(params):
    """Fonction objectif optimisée"""
    current_params = [
        params[0][0],          # lr
        int(params[0][1]),     # filters
        int(params[0][2]),     # dense_units
        params[0][3],          # dropout
        params[0][4]           # l2
    ]

    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

    # Réduction des données
    x_train = x_train[:3000].reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = y_train[:3000]
    x_val = x_val.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    model = build_model([current_params])

    # Entraînement
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=[
            callbacks.EarlyStopping(patience=2),
            callbacks.ModelCheckpoint("temp_model.keras", save_best_only=True)
        ],
        verbose=1
    )

    return history.history['val_loss'][-1]

# Espace de recherche
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
    {'name': 'filters', 'type': 'discrete', 'domain': (16, 32)},
    {'name': 'dense_units', 'type': 'discrete', 'domain': (64, 128)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.1, 0.3)},
    {'name': 'l2', 'type': 'continuous', 'domain': (1e-5, 1e-3)}
]

optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective_function,
    domain=bounds,
    acquisition_type='EI',
    exact_feval=True,
    maximize=False,
    initial_design_numdata=3
)

print("Début de l'optimisation...")
optimizer.run_optimization(max_iter=30)  # Exécute 30 itérations

print("Optimisation terminée !")

# Personnalisation du graphique de convergence
plt.figure(figsize=(10, 6))
plt.plot(optimizer.Y, label='Perte de Validation')
plt.xlabel('Itération')
plt.ylabel('Perte de Validation')
plt.title('Convergence de l\'Optimisation Bayésienne')
plt.legend()
plt.grid(True)
plt.savefig('convergence.png')
plt.show()

# Sauvegarde du rapport d'optimisation
with open('bayes_opt.txt', 'w') as report_file:
    report_file.write("Rapport d'Optimisation Bayésienne\n")
    report_file.write("================================\n")
    best_params = optimizer.X[np.argmin(optimizer.Y)]
    report_file.write(f"Meilleur taux d'apprentissage : {best_params[0]:.4f}\n")
    report_file.write(f"Meilleur nombre de filtres : {int(best_params[1])}\n")
    report_file.write(f"Meilleur nombre d'unités denses : {int(best_params[2])}\n")
    report_file.write(f"Meilleur taux de dropout : {best_params[3]:.2f}\n")
    report_file.write(f"Meilleur poids de régularisation L2 : {best_params[4]:.4f}\n")
    report_file.write(f"Meilleure perte de validation : {np.min(optimizer.Y):.4f}\n")
    report_file.write("\nHistorique des Itérations:\n")
    report_file.write("==========================\n")
    for i, (params, loss) in enumerate(zip(optimizer.X, optimizer.Y)):
        report_file.write(f"Iteration {i+1}:\n")
        report_file.write(f"  Taux d'apprentissage : {params[0]:.4f}\n")
        report_file.write(f"  Nombre de filtres : {int(params[1])}\n")
        report_file.write(f"  Nombre d'unités denses : {int(params[2])}\n")
        report_file.write(f"  Taux de dropout : {params[3]:.2f}\n")
        report_file.write(f"  Poids de régularisation L2 : {params[4]:.4f}\n")
        report_file.write(f"  Perte de validation : {loss[0]:.4f}\n")
        report_file.write("--------------------------\n")

print("Rapport d'optimisation sauvegardé dans 'bayes_opt.txt'")