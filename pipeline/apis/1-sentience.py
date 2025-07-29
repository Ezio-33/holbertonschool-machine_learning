#!/usr/bin/env python3
"""
Module pour récupérer les planètes d'origine des espèces sensibles
via l'API Swapi
"""
import requests


def sentientPlanets():
    """
    Retourne la liste des noms des planètes d'origine de toutes les espèces
    sensibles

    Returns:
        list: Liste des noms des planètes d'origine des espèces sensibles
    """
    planets = []
    species_url = "https://swapi-api.alx-tools.com/api/species/"

    while species_url:
        try:
            response = requests.get(species_url)
            response.raise_for_status()
            data = response.json()

            for species in data.get('results', []):
                classification = species.get('classification', '').lower()
                designation = species.get('designation', '').lower()

                # Vérifier si l'espèce est sensible
                if 'sentient' in classification or 'sentient' in designation:
                    homeworld_url = species.get('homeworld')
                    if homeworld_url:
                        try:
                            planet_response = requests.get(homeworld_url)
                            planet_response.raise_for_status()
                            planet_data = planet_response.json()
                            planet_name = planet_data.get('name')
                            if planet_name and planet_name not in planets:
                                planets.append(planet_name)
                        except requests.exceptions.RequestException:
                            continue

            # Gérer la pagination
            species_url = data.get('next')

        except requests.exceptions.RequestException:
            break

    return planets
