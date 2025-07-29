#!/usr/bin/env python3
"""
Script pour afficher le nombre de lancements par fusée en utilisant
l'API SpaceX
"""
import requests
from collections import defaultdict


def get_rocket_frequency():
    """
    Récupère et affiche le nombre de lancements par fusée
    """
    try:
        # Récupérer tous les lancements
        launches_response = requests.get(
            "https://api.spacexdata.com/v4/launches")
        launches_response.raise_for_status()
        launches = launches_response.json()

        # Récupérer toutes les fusées
        rockets_response = requests.get(
            "https://api.spacexdata.com/v4/rockets")
        rockets_response.raise_for_status()
        rockets = rockets_response.json()

        # Créer un dictionnaire rocket_id -> rocket_name
        rocket_names = {rocket['id']: rocket['name'] for rocket in rockets}

        # Compter les lancements par fusée
        rocket_counts = defaultdict(int)

        for launch in launches:
            rocket_id = launch.get('rocket')
            if rocket_id in rocket_names:
                rocket_name = rocket_names[rocket_id]
                rocket_counts[rocket_name] += 1

        # Trier par nombre de lancements (décroissant) puis par ordre
        # alphabétique
        sorted_rockets = sorted(rocket_counts.items(),
                                key=lambda x: (-x[1], x[0]))

        # Afficher les résultats
        for rocket_name, count in sorted_rockets:
            print(f"{rocket_name}: {count}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    except Exception as e:
        print(f"Error processing data: {e}")


if __name__ == "__main__":
    get_rocket_frequency()
