#!/usr/bin/env python3
"""
Script qui crée un pd.DataFrame à partir d'un dictionnaire.
"""

import pandas as pd

# Créer le dictionnaire avec les données spécifiées
data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Créer le DataFrame avec les index spécifiés
df = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])
