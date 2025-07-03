# Deep Q-Learning - Atari Breakout

## Installation

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Installer les ROMs Atari

```bash
AutoROM --accept-license
```

### 3. Correction manuelle nécessaire pour keras-rl2

**IMPORTANT**: Après l'installation, vous devez modifier le fichier callbacks.py de keras-rl2 :

**Fichier à modifier :**

```bash
# Trouver le fichier (généralement dans votre environnement virtuel)
find ~/.pyenv/versions/*/lib/python*/site-packages/rl -name "callbacks.py"
```

**Modification à apporter :**

Trouvez la ligne 8 dans `callbacks.py` :

```python
from tensorflow.keras import __version__ as KERAS_VERSION
```

Et remplacez-la par :

```python
try:
    from tensorflow.keras import __version__ as KERAS_VERSION
except ImportError:
    KERAS_VERSION = '2.10.0'
```

### 4. Utilisation

**Entraînement :**

```bash
python3 train.py
```

**Test de l'agent entraîné :**

```bash
python3 play.py
```

**Test avec agent aléatoire (comparaison) :**

```bash
python3 test_random.py
```

## Notes importantes

- L'entraînement complet prend plusieurs heures (1M d'étapes)
- Un entraînement court (30k étapes) ne donnera pas de bons résultats
- Les versions dans requirements.txt sont testées et compatibles
- La modification de callbacks.py est nécessaire pour éviter l'erreur ImportError

## Versions alternatives

Si vous rencontrez des problèmes, vous pouvez aussi essayer :

```bash
pip uninstall tensorflow keras keras-rl2 -y
pip install tensorflow==2.11.0 keras-rl2==1.0.5
```
