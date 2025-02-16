# -*- coding: utf-8 -*-
# Author : Badr TAJINI
#
'''
# Code Python très simple pour construire un modèle MLP basique
#
# Objectifs de l'Étape 2 :
# 
# 1- Chargement de Données : Introduire l'idée de charger un dataset standard (MNIST) en utilisant scikit-learn.
# 2- Forme des Données : Comprendre la structure des données d'images (nombre d'échantillons, nombre de caractéristiques).
# 3- Construction d'un Modèle : Apprendre à instancier un modèle MLP de base en utilisant MLPClassifier de scikit-learn.=
# 4- Hyperparamètres : Être initié au concept des hyperparamètres d'un modèle (comme le nombre de couches cachées et le nombre de neurones par couche) et comment les modifier lors de la construction du modèle.
# 
# Note : Pas d'Entraînement (pour l'instant) : Se concentrer sur la construction du modèle, la prochaine étape se concentrera sur l'entraînement.
# 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Instructions :
#
# 1- Exécutez le code.
# 2- Observez les sorties de print :
#    - Que montrent les formes des données X et y ? (Nombre d'images, nombre de pixels par image, nombre d'étiquettes)
#    - Que se passe-t-il lorsque vous créez différentes instances de MLPClassifier ? Quels sont les paramètres que vous pouvez modifier ? (Par exemple, hidden_layer_sizes, solver)
#
# 3- Expérimentez :
#    - Modifiez les valeurs de hidden_layer_sizes. Que se passe-t-il si vous mettez une seule valeur ? Deux valeurs ?
#    - Essayez de changer le solver par 'sgd'.
#
# References : 
# - Visualisation d'un MLPClassifier : https://scikit-learn.org/1.5/auto_examples/neural_networks/plot_mnist_filters.html
'''


from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import numpy as np

# 1. Charger le dataset MNIST
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target

# 2. Examiner les données (pour comprendre leur forme)
print("Forme des données d'images (X) :", X.shape)
print("Forme des étiquettes (y) :", y.shape)

# 3. Préparer les données (une étape simple : mise à l'échelle - peut être simplifiée au début)
#    Note : Pour simplifier au maximum, on pourrait même sauter cette étape au début.
X = X / 255.0

# 4. Construire un modèle MLP basique
#    Ici, nous allons juste instancier le modèle avec différents paramètres.

# Exemple 1 : Un MLP très simple avec une seule couche cachée de 50 neurones
mlp_simple = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10)
print("Modèle MLP simple créé :", mlp_simple)

# Exemple 2 : Un MLP avec deux couches cachées
mlp_deux_couches = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=10)
print("Modèle MLP à deux couches créé :", mlp_deux_couches)

# Exemple 3 : Essayer différents algorithmes d'optimisation
mlp_adam = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', max_iter=10)
print("MLP avec optimiseur Adam :", mlp_adam)

mlp_sgd = MLPClassifier(hidden_layer_sizes=(50,), solver='sgd', learning_rate_init=0.01, max_iter=10)
print("MLP avec optimiseur SGD :", mlp_sgd)

# Remarque : `max_iter` est limité ici pour éviter que l'entraînement ne prenne trop de temps si on l'exécute.
# L'objectif principal est l'instanciation du modèle.