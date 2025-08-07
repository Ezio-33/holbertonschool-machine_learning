#!/usr/bin/env python3
"""
Fonction qui change tous les sujets d'un document école basé sur le nom
"""


def update_topics(mongo_collection, name, topics):
    """
    Change tous les sujets d'un document école basé sur le nom

    Args:
        mongo_collection: objet collection pymongo
        name (string): nom de l'école à mettre à jour
        topics (list of strings): liste des sujets abordés dans l'école
    """
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )
