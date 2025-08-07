#!/usr/bin/env python3
"""
Script qui fournit des statistiques sur les logs Nginx stockés dans MongoDB
"""
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs_collection = client.logs.nginx

    # Compte le total des logs
    total_logs = logs_collection.count_documents({})
    print(f"{total_logs} logs")

    # Compte les méthodes
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        count = logs_collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")

    # Compte les vérifications de statut
    status_check = logs_collection.count_documents({
        "method": "GET",
        "path": "/status"
    })
    print(f"{status_check} status check")
