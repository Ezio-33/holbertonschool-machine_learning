# Databases

## Description

Ce projet explore les bases de données relationnelles et non-relationnelles, essentielles pour le stockage de données dans le cadre du Machine Learning.

## Objectifs d'apprentissage

- Comprendre les bases de données relationnelles vs non-relationnelles
- Maîtriser SQL et NoSQL
- Créer des tables avec contraintes
- Optimiser les requêtes avec des index
- Implémenter des procédures stockées, fonctions, vues et triggers en MySQL
- Comprendre ACID et le stockage de documents
- Utiliser MongoDB pour les opérations CRUD
- Intégrer Python avec MongoDB via PyMongo

## Technologies utilisées

- **MySQL 8.0** : Base de données relationnelle
- **MongoDB 4.4** : Base de données NoSQL
- **Python 3.9** : Scripts d'interaction avec MongoDB
- **PyMongo 4.6.2** : Driver MongoDB pour Python

## Structure du projet

### Partie MySQL (Tâches 0-21)

- **Bases** : Création de bases de données, tables, insertions
- **Requêtes** : SELECT avec conditions, tri, agrégation
- **Avancé** : Index, triggers, procédures stockées, fonctions, vues

### Partie MongoDB (Tâches 22-34)

- **Bases** : Création de bases, collections, documents
- **Opérations CRUD** : Create, Read, Update, Delete
- **Python intégration** : Scripts PyMongo pour opérations avancées

### Partie Avancée (Tâches 35-41)

- **Optimisation** : Index composites, vues complexes
- **Analytics** : Agrégation de données, statistiques

## Installation et Configuration

### MySQL

```bash
sudo apt-get update
sudo apt-get install mysql-server
mysql -uroot -p
```

### MongoDB

```bash
wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-4.4.list
sudo apt-get update
sudo apt-get install -y mongodb-org
pip3 install pymongo
```

## Utilisation

Chaque fichier correspond à une tâche spécifique. Les fichiers SQL peuvent être exécutés avec :

```bash
cat fichier.sql | mysql -hlocalhost -uroot -p [nom_base]
```

Les scripts MongoDB avec :

```bash
cat fichier | mongo [nom_base]
```

Les scripts Python avec :

```bash
python3 fichier.py
```
