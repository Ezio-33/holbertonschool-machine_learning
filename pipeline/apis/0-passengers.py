#!/usr/bin/env python3
"""
Module pour récupérer les vaisseaux disponibles via l'API Swapi
"""
import requests


def availableShips(passengerCount):
    """
    Retourne la liste des vaisseaux pouvant accueillir un nombre donné
    de passagers

    Args:
        passengerCount (int): Nombre minimum de passagers que le vaisseau
                              doit pouvoir accueillir

    Returns:
        list: Liste des noms des vaisseaux disponibles
    """
    ships = []
    url = "https://swapi-api.alx-tools.com/api/starships/"

    while url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            for ship in data.get('results', []):
                passengers = ship.get('passengers', '0')
                # Gérer les cas où passengers contient des caractères
                # non numériques
                if passengers == 'n/a' or passengers == 'unknown':
                    continue

                # Nettoyer la chaîne (supprimer les virgules et autres
                # caractères)
                passengers_clean = passengers.replace(',', '').replace(
                    '-', '').strip()

                try:
                    passengers_int = int(passengers_clean)
                    if passengers_int >= passengerCount:
                        ships.append(ship.get('name'))
                except ValueError:
                    # Si la conversion échoue, ignorer ce vaisseau
                    continue

            # Gérer la pagination
            url = data.get('next')

        except requests.exceptions.RequestException:
            # En cas d'erreur de requête, arrêter et retourner ce qu'on a
            break

    return ships
