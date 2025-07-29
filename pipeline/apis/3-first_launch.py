#!/usr/bin/env python3
"""
Script pour afficher le premier lancement spatial en utilisant l'API SpaceX
"""
import requests
from datetime import datetime


def get_first_launch():
    """
    Récupère et affiche les informations du premier lancement spatial
    """
    try:
        # Récupérer tous les lancements
        launches_response = requests.get(
            "https://api.spacexdata.com/v4/launches")
        launches_response.raise_for_status()
        launches = launches_response.json()

        if not launches:
            print("No launches found")
            return

        # Trier par date_unix (le plus ancien en premier)
        launches.sort(key=lambda x: x.get('date_unix', 0))
        first_launch = launches[0]

        # Récupérer les détails de la fusée
        rocket_id = first_launch.get('rocket')
        rocket_response = requests.get(
            f"https://api.spacexdata.com/v4/rockets/{rocket_id}")
        rocket_response.raise_for_status()
        rocket_data = rocket_response.json()
        rocket_name = rocket_data.get('name')

        # Récupérer les détails du launchpad
        launchpad_id = first_launch.get('launchpad')
        launchpad_response = requests.get(
            f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}")
        launchpad_response.raise_for_status()
        launchpad_data = launchpad_response.json()
        launchpad_name = launchpad_data.get('name')
        launchpad_locality = launchpad_data.get('locality')

        # Formater la date
        date_local = first_launch.get('date_local')
        launch_name = first_launch.get('name')

        # Afficher le résultat
        print(f"{launch_name} ({date_local}) {rocket_name} - "
              f"{launchpad_name} ({launchpad_locality})")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    except Exception as e:
        print(f"Error processing data: {e}")


if __name__ == "__main__":
    get_first_launch()
