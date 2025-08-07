#!/usr/bin/env python3
"""
Fonction qui liste tous les documents dans une collection
"""


def list_all(mongo_collection):
    """
    Liste tous les documents dans une collection

    Args:
        mongo_collection: objet collection pymongo

    Returns:
        Liste des documents ou liste vide si aucun document dans la collection
    """
    return list(mongo_collection.find())
