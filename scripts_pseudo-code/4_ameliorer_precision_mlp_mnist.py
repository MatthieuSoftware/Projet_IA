# -*- coding: utf-8 -*-
# Author : Badr TAJINI
#
'''
# Code Python très simple pour construire un modèle MLP basique avec l'amélioration de la précision
#
# Objectifs de l'Étape 4 :
#
# 1- Impact de l'Architecture : Comprendre comment la complexité du réseau neuronal (nombre de couches et de neurones) influence sa capacité à apprendre et sa précision.
# 2- Régularisation : Introduire le concept de régularisation comme moyen de prévenir le sur-apprentissage et d'améliorer la généralisation du modèle.
# 3- Algorithmes d'Optimisation : Explorer différents algorithmes d'optimisation et comprendre que le choix de l'optimiseur et de ses paramètres (comme le taux d'apprentissage) peut avoir un impact significatif sur l'entraînement et les performances du modèle.
# 4- Hyperparamètres et Ajustement : Renforcer l'idée que la construction d'un bon modèle implique de choisir et d'ajuster les bons hyperparamètres.
# 5- Évaluation Comparative : Apprendre à comparer les performances de différents modèles et configurations.
# 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Instructions :
#
# 1- Exécutez le code.
# 2- Observez les sorties :
#    - Comment la précision change-t-elle lorsque vous modifiez l'architecture du modèle (plus de neurones, plus de couches) ?
#    - Quel est l'impact de l'ajout de la régularisation L2 ? Comprenez-vous l'idée de pénaliser les poids importants pour éviter le sur-apprentissage ?
#    - Comment les différents algorithmes d'optimisation (adam vs sgd) affectent-ils la précision ?
# 3- Expérimentez :
#    - Architecture : Essayez différentes configurations pour hidden_layer_sizes. Par exemple, (50, 50), (150,), (128, 64, 32). Y a-t-il une limite au nombre de couches ou de neurones que vous pouvez ajouter ?
#    - Régularisation : Modifiez la valeur du paramètre alpha. Qu'arrive-t-il à la précision si vous augmentez ou diminuez alpha ?
#    - Optimisation :
#      - Pour l'optimiseur sgd, essayez différents taux d'apprentissage (learning_rate_init). Un taux plus élevé permet-il d'apprendre plus vite ? Est-ce toujours bénéfique ?
#      - Recherchez d'autres optimiseurs disponibles dans MLPClassifier (par exemple, lbfgs, bien que moins adapté aux grands datasets) et testez-les.
#    - Nombre d'itérations : Si vos expériences sont rapides, essayez d'augmenter max_iter pour voir si le modèle continue de s'améliorer. Soyez patients, l'entraînement peut prendre plus de temps.
'''

from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Charger et préparer le dataset MNIST (comme à l'étape 3)
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Explorer différentes architectures de modèles MLP
print("\n--- Exploration de différentes architectures ---")

# Modèle 1 : Plus de neurones dans une seule couche cachée
mlp_large = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, random_state=42)
mlp_large.fit(X_train, y_train)
y_pred_large = mlp_large.predict(X_test)
accuracy_large = accuracy_score(y_test, y_pred_large)
print(f"Précision avec une couche cachée de 100 neurones : {accuracy_large * 100:.2f}%")

# Modèle 2 : Plusieurs couches cachées
mlp_multi = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=10, random_state=42)
mlp_multi.fit(X_train, y_train)
y_pred_multi = mlp_multi.predict(X_test)
accuracy_multi = accuracy_score(y_test, y_pred_multi)
print(f"Précision avec deux couches cachées (100, 50 neurones) : {accuracy_multi * 100:.2f}%")

# 3. Introduction à la régularisation (L2) pour éviter le sur-apprentissage
print("\n--- Introduction à la régularisation ---")

# Modèle 3 : Avec régularisation L2 (paramètre alpha)
mlp_regularized = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=0.001, random_state=42)
mlp_regularized.fit(X_train, y_train)
y_pred_regularized = mlp_regularized.predict(X_test)
accuracy_regularized = accuracy_score(y_test, y_pred_regularized)
print(f"Précision avec régularisation L2 (alpha=0.001) : {accuracy_regularized * 100:.2f}%")

# 4. Explorer différents algorithmes d'optimisation
print("\n--- Exploration de différents algorithmes d'optimisation ---")

# Modèle 4 : Utilisation de l'optimiseur 'adam' (qui est l'optimiseur par défaut)
mlp_adam = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, solver='adam', random_state=42)
mlp_adam.fit(X_train, y_train)
y_pred_adam = mlp_adam.predict(X_test)
accuracy_adam = accuracy_score(y_test, y_pred_adam)
print(f"Précision avec l'optimiseur Adam : {accuracy_adam * 100:.2f}%")

# Modèle 5 : Utilisation de l'optimiseur 'sgd' (Stochastic Gradient Descent) avec un taux d'apprentissage
mlp_sgd = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, solver='sgd', learning_rate_init=0.01, random_state=42)
mlp_sgd.fit(X_train, y_train)
y_pred_sgd = mlp_sgd.predict(X_test)
accuracy_sgd = accuracy_score(y_test, y_pred_sgd)
print(f"Précision avec l'optimiseur SGD (taux d'apprentissage=0.01) : {accuracy_sgd * 100:.2f}%")

# Remarque : `max_iter` est toujours limité ici pour des raisons de temps d'exécution lors des tests.
# Pour obtenir de meilleures performances, il faudrait augmenter le nombre d'itérations.