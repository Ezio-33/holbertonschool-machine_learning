#!/usr/bin/env python3
"""
Fonction qui insère un nouveau document dans une collection basé sur kwargs
"""


def insert_school(mongo_collection, **kwargs):
    """
    Insère un nouveau document dans une collection basé sur kwargs

    Args:
        mongo_collection: objet collection pymongo
        **kwargs: champs du document

    Returns:
        Le nouvel _id
    """
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
