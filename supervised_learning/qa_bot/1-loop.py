#!/usr/bin/env python3
"""
Boucle de chat Q&R simple qui demande continuellement
les questions de l'utilisateur
et fournit des réponses jusqu'à ce qu'une commande de sortie soit reçue.

Ce module représente un cadre de base pour un bot de questions-réponses qui:
1. Invite l'utilisateur à saisir une entrée préfixée par "Q: "
2. Quitte gracieusement lorsque l'utilisateur fournit
une des commandes de sortie ('exit', 'quit', 'goodbye', 'bye')
3. Sinon répond avec une réponse (actuellement vide)

Utilisation:
      Exécutez le script directement pour démarrer la session interactive Q&R.
      Tapez 'exit', 'quit', 'goodbye', ou 'bye' pour terminer la session.
"""

while True:
    val = input("Q: ")
    exit_list = ['exit', 'quit', 'goodbye', 'bye']
    if val.lower() in exit_list:
        print("A: Goodbye")
        break
    answer = ''
    print("A: {}".format(answer))
