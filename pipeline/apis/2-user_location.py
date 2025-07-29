#!/usr/bin/env python3
"""
Script pour afficher la localisation d'un utilisateur GitHub
en gérant les limites de taux
"""
import requests
import sys
from datetime import datetime


def get_user_location(api_url):
    """
    Récupère la localisation d'un utilisateur GitHub

    Args:
        api_url (str): URL de l'API GitHub pour l'utilisateur
    """
    try:
        response = requests.get(api_url)

        if response.status_code == 200:
            user_data = response.json()
            location = user_data.get('location')
            if location:
                print(location)
            else:
                print("Not found")
        elif response.status_code == 404:
            print("Not found")
        elif response.status_code == 403:
            # Gérer la limite de taux
            reset_time = response.headers.get('X-RateLimit-Reset')
            if reset_time:
                reset_timestamp = int(reset_time)
                current_timestamp = int(datetime.now().timestamp())
                minutes_to_reset = (reset_timestamp - current_timestamp) // 60
                if minutes_to_reset > 0:
                    print(f"Reset in {minutes_to_reset} min")
                else:
                    print("Reset in 1 min")
            else:
                print("Reset in 60 min")
        else:
            print("Not found")

    except requests.exceptions.RequestException:
        print("Not found")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <github_api_url>")
        sys.exit(1)

    api_url = sys.argv[1]
    get_user_location(api_url)
