# -*- coding: utf-8 -*-
# Author : Badr TAJINI
#
'''
# Code Python très simple pour classifier des fruits
#
# Objectifs de l'Étape 1 :
1- Exécution de Code : Se familiariser avec l'exécution d'un script Python simple.
2- Observation des Résultats : Comprendre que le code prend des entrées (caractéristiques des fruits) et produit des sorties (prédictions de fruits).
3- Introduction à un Modèle ML : Voir un exemple concret de l'utilisation d'un modèle d'apprentissage automatique (ici, un arbre de décision) pour faire des prédictions.
4- Concept d'Entraînement : Comprendre que le modèle a besoin d'être "entraîné" avec des données existantes avant de pouvoir faire des prédictions sur de nouvelles données.
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Instructions :
#
# 1- Exécutez le code.
# 2- Observez les résultats. Qu'est-ce que le programme affiche ?
# 3- Expérimentez (facultatif, mais encouragé) :
#    - Modifiez les caractéristiques des nouveaux_fruits. Quelles sont les nouvelles prédictions ?
#    - Ajoutez d'autres exemples de fruits dans les attributs et etiquettes. Exécutez à nouveau le code. Est-ce que les prédictions changent ?
'''

from sklearn.tree import DecisionTreeClassifier

# Nos données de fruits (simplifiées)
# Chaque sous-liste représente un fruit avec ses caractéristiques : [couleur, forme]
attributs = [
    ["Rouge", "Ronde"],
    ["Jaune", "Allongée"],
    ["Orange", "Ronde"],
    ["Vert", "Ronde"],
    ["Jaune", "Ronde"]
    ["orange",""]
]
etiquettes = ["Pomme", "Banane", "Orange", "Pomme", "Banane"] # Les noms des fruits correspondants

# Créer un modèle d'arbre de décision (un modèle simple de classification)
modele = DecisionTreeClassifier()

# Apprendre au modèle à partir des données (l'entraîner)
modele.fit(attributs, etiquettes)

# Faire des prédictions pour de nouveaux fruits
nouveaux_fruits = [
    ["Rouge", "Ronde"],
    ["Jaune", "Allongée"],
    ["Vert", "Ronde"]
]
predictions = modele.predict(nouveaux_fruits)

# Afficher les prédictions
print("Prédictions pour les nouveaux fruits :")
for i in range(len(nouveaux_fruits)):
    print(f"Un fruit {nouveaux_fruits[i][0]} et {nouveaux_fruits[i][1]} est prédit comme étant un(e) : {predictions[i]}")
    
    
    
