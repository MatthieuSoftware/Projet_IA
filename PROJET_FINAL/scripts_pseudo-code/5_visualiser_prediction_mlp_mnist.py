# -*- coding: utf-8 -*-
# Author : Badr TAJINI
#
'''
# Code Python très simple pour construire un modèle MLP basique avec la visualisation de la prédiction
#
# Objectifs de l'Étape 5 :
#
# 1- Visualisation des Prédictions : Obtenir une compréhension visuelle des performances du modèle en examinant directement les images et leurs prédictions.
# 2- Matrice de Confusion : Apprendre à interpréter une matrice de confusion pour analyser en détail les types d'erreurs que le modèle commet pour chaque classe.
# 3- Évaluation Qualitative : Comprendre que l'évaluation des modèles ne se limite pas aux métriques numériques comme la précision, mais peut également inclure une analyse qualitative des résultats.
# 4- Corrélation entre Métriques et Visualisations : Faire le lien entre la précision globale du modèle et les informations visuelles fournies par les prédictions et la matrice de confusion. Un modèle avec une précision plus élevée devrait avoir moins d'erreurs visibles et une matrice de confusion avec une diagonale plus forte.
# 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Instructions :
#
# 1- Exécutez le code  (Assurez-vous que les librairies matplotlib et seaborn sont installées).
# 2- Observez les visualisations :
#    - Prédictions sur les images : Regardez les 10 premières images de test. Le modèle a-t-il correctement prédit le chiffre ? Y a-t-il des cas où la prédiction est incorrecte ? Essayez de deviner pourquoi le modèle a pu se tromper dans ces cas (par exemple, un chiffre mal écrit).
#    - Matrice de Confusion : Analysez la matrice de confusion.
#      - Que représente chaque ligne et chaque colonne ?
#      - Les valeurs élevées se trouvent-elles principalement sur la diagonale ? Qu'est-ce que cela signifie ?
#      - Y a-t-il des zones où les valeurs sont plus élevées en dehors de la diagonale ? Quels types d'erreurs le modèle fait-il le plus souvent (par exemple, confond-il souvent le 4 et le 9) ?
# 3- Expérimentez :
#    - Modifiez le modèle : Réutilisez les modèles que vous avez entraînés à l'étape 4 (avec différentes architectures, régularisation, etc.). Comment les visualisations changent-elles en fonction des performances du modèle ? Un modèle plus précis a-t-il une matrice de confusion avec une diagonale plus "prononcée" ?
#    - Affichez plus d'images : Modifiez la boucle pour afficher plus de 10 images. Voyez-vous plus d'erreurs ?
#    - Examinez les erreurs spécifiques : Essayez d'identifier des images où le modèle s'est trompé (en comparant y_test_pred et y_test) et affichez ces images pour essayer de comprendre la source de l'erreur.
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 1. Charger et préparer le dataset MNIST (comme dans les étapes précédentes)
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. (Rappel) Créer et entraîner le modèle (vous pouvez réutiliser le meilleur modèle trouvé à l'étape 4)
#    Ici, on utilise un modèle simple pour l'exemple. N'hésitez pas à encourager les étudiants
#    à utiliser les modèles qu'ils ont entraînés précédemment.
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, random_state=42)
model.fit(X_train, y_train)

# 3. Obtenir les probabilités prédites pour l'ensemble de test
#    Pour chaque image de test, le modèle donne une probabilité d'appartenance à chaque classe (chiffre de 0 à 9).
y_test_probs = model.predict_proba(X_test)
print("Probabilités prédites pour la première image de test :\n", y_test_probs[0])

# 4. Convertir les probabilités en prédictions de classe
#    On choisit la classe avec la probabilité la plus élevée comme prédiction.
y_test_pred = np.argmax(y_test_probs, axis=1)
print("\nClasses prédites (les 20 premières) :", y_test_pred[:20])
print("Classes réelles     (les 20 premières) :", y_test[:20].values[:20]) # Accès aux valeurs NumPy

# 5. Visualiser les prédictions sur les 10 premières images de test
plt.figure(figsize=(20, 4))
for index in range(10):
    plt.subplot(2, 5, index + 1)
    # Remodeler l'image aplatie en une image 28x28 pour l'affichage
    plt.imshow(X_test.iloc[index].values.reshape(28, 28), cmap=plt.cm.gray) # Accès aux valeurs NumPy
    plt.title(f"Prédit : {y_test_pred[index]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# 6. Matrice de Confusion
#    Affiche le nombre de fois où chaque chiffre a été confondu avec un autre.
#    La normalisation permet de voir les proportions.
cm = confusion_matrix(y_test, y_test_pred, normalize='true')
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap="Blues_r")
plt.ylabel('Vraie étiquette')
plt.xlabel('Étiquette prédite')
plt.title('Matrice de Confusion (Normalisée)', size=15)
plt.show()

# 7. Afficher la précision globale (pour comparer avec les visualisations)
accuracy = np.mean(y_test_pred == y_test)
print(f"\nPrécision du modèle sur l'ensemble de test : {accuracy * 100:.2f}%")