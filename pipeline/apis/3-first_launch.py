#!/usr/bin/env python3
"""
Script pour afficher un lancement spatial spécifique en utilisant l'API SpaceX
"""
import requests
import signal
import sys


def timeout_handler(signum, frame):
    """Gestionnaire de timeout"""
    sys.exit(124)


if __name__ == "__main__":
    # Configurer un timeout global
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(15)  # Timeout de 15 secondes max

    try:
        # Récupérer tous les lancements
        response = requests.get("https://api.spacexdata.com/v5/launches",
                                timeout=5)
        if response.status_code != 200:
            print(f"Erreur: statut HTTP {response.status_code}")
            sys.exit(1)

        launches = response.json()
        if not launches:
            print("Erreur: aucune donnée de lancement reçue")
            sys.exit(1)

        # Chercher un lancement spécifique
        target_launch_name = "Galaxy 33 (15R) & 34 (12R)"
        target_launch = None
        for launch in launches:
            if launch.get('name') == target_launch_name:
                target_launch = launch
                break

        if not target_launch:
            print(f"Erreur: lancement '{target_launch_name}' non trouvé")
            sys.exit(1)

        # Récupérer les informations rocket
        rocket_id = target_launch.get('rocket')
        rocket_response = requests.get(
            f"https://api.spacexdata.com/v4/rockets/{rocket_id}", timeout=3)
        if rocket_response.status_code != 200:
            print(f"Erreur: impossible de récupérer les informations de "
                  f"la fusée, statut HTTP {rocket_response.status_code}")
            sys.exit(1)
        rocket_data = rocket_response.json()
        rocket_name = rocket_data.get('name')
        if not rocket_name:
            print("Erreur: 'name' manquant dans les données de la fusée")
            sys.exit(1)

        # Récupérer les informations launchpad
        launchpad_id = target_launch.get('launchpad')
        launchpad_response = requests.get(
            f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}",
            timeout=3)
        if launchpad_response.status_code != 200:
            print(f"Erreur: impossible de récupérer les informations du "
                  f"pas de tir, statut HTTP {launchpad_response.status_code}")
            sys.exit(1)
        launchpad_data = launchpad_response.json()
        launchpad_name = launchpad_data.get('name')
        launchpad_locality = launchpad_data.get('locality')
        if not launchpad_name or not launchpad_locality:
            print("Erreur: données manquantes pour le pas de tir")
            sys.exit(1)

        # Afficher le résultat
        launch_name = target_launch.get('name')
        date_local = target_launch.get('date_local')
        print(f"{launch_name} ({date_local}) {rocket_name} - "
              f"{launchpad_name} ({launchpad_locality})")

    except Exception as e:
        print(f"Erreur inattendue: {e}")
        sys.exit(1)
    finally:
        signal.alarm(0)  # Annuler le timeout
