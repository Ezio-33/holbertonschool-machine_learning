#!/usr/bin/env python3
"""
Fonction qui retourne la liste des écoles ayant un sujet spécifique
"""


def schools_by_topic(mongo_collection, topic):
    """
    Retourne la liste des écoles ayant un sujet spécifique

    Args:
        mongo_collection: objet collection pymongo
        topic (string): sujet recherché

    Returns:
        Liste des écoles avec le sujet spécifique
    """
    return list(mongo_collection.find({"topics": topic}))
