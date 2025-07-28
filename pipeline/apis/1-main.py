#!/usr/bin/env python3
"""
Test file for 1-sentience.py
"""
sentientPlanets = __import__('1-sentience').sentientPlanets
planets = sentientPlanets()
for planet in planets:
    print(planet)
