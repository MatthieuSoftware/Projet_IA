# -*- coding: utf-8 -*-
# Author : Badr TAJINI
#
'''
# Code Python très simple pour construire un modèle MLP basique avec un entrainement et une évaluation
#
# Objectifs de l'Étape 3 :
#
# 1- Division des Données : Comprendre l'importance de diviser les données en ensembles d'entraînement et de test pour une évaluation réaliste du modèle.
# 2- Entraînement d'un Modèle : Voir concrètement comment un modèle MLP est entraîné en utilisant la méthode fit().
# 3- Prédiction : Apprendre à utiliser la méthode predict() pour faire des prédictions sur de nouvelles données.
# 4- Évaluation de la Précision : Comprendre ce que représente la précision et comment elle est calculée en utilisant accuracy_score.
# 5- Impact des Hyperparamètres : Commencer à explorer l'impact de certains hyperparamètres (comme le nombre d'itérations et l'architecture du réseau) sur les performances du modèle.
# 6- Le Processus de l'Apprentissage Automatique : Avoir une vue d'ensemble du processus : chargement des données, préparation, construction du modèle, entraînement, prédiction et évaluation.
# 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Instructions :
#
# 1- Exécutez le code.
# 2- Observez les sorties :
#    - Combien de temps prend l'entraînement ?
#    - Quelle est la précision du modèle sur l'ensemble de test ? Comprenez-vous ce que représente ce pourcentage ?
#    - Regardez les exemples de prédictions et les étiquettes réelles. Le modèle a-t-il fait des erreurs sur ces exemples ?
# 3- Expérimentez :
#    - Modifiez le nombre d'itérations (max_iter) : Augmentez max_iter à 20 ou 50. Relancez le code. La précision s'améliore-t-elle ? L'entraînement prend-il plus de temps ?
#    - Modifiez l'architecture du modèle (hidden_layer_sizes) : Essayez hidden_layer_sizes=(100,) ou hidden_layer_sizes=(100, 50). Comment cela affecte-t-il la précision et le temps d'entraînement ?
#    - Examinez la division des données : Modifiez test_size dans train_test_split. Quel est l'impact sur la quantité de données utilisées pour l'entraînement et le test ?
'''

from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Charger le dataset MNIST
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype(np.float32) / 255.0  # Normaliser les données
y = mnist.target.astype(int)

# 2. Diviser les données en ensembles d'entraînement et de test
#    C'est crucial pour évaluer les performances du modèle sur des données qu'il n'a jamais vues.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Construire un modèle MLP (vous pouvez utiliser un des modèles de l'étape 2 ou en créer un nouveau)
#    Commençons par un modèle simple pour l'entraînement initial.
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, random_state=42)
print("Modèle MLP créé :", mlp)

# 4. Entraîner le modèle sur les données d'entraînement
#    C'est là que le modèle apprend à reconnaître les chiffres.
print("\nDébut de l'entraînement du modèle...")
mlp.fit(X_train, y_train)
print("Entraînement terminé.")

# 5. Faire des prédictions sur l'ensemble de test
#    Utiliser le modèle entraîné pour prédire les chiffres sur l'ensemble de test.
y_pred = mlp.predict(X_test)

# 6. Évaluer les performances du modèle en calculant la précision
#    La précision mesure le pourcentage de chiffres correctement classifiés.
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrécision du modèle sur l'ensemble de test : {accuracy * 100:.2f}%")

# 7. (Facultatif) Afficher quelques prédictions et les étiquettes réelles pour comparaison
print("\nQuelques prédictions et étiquettes réelles :")
for i in range(10):
    print(f"Image {i+1}: Prédiction = {y_pred[i]}, Réel = {y_test.iloc[i]}")