<div id="top">

<!-- STYLE D'EN-TÊTE : MODERNE -->
<div align="left" style="position: relative; width: 100%; height: 100%;">

	<img src="readmeai/assets/logos/purple.svg" width="30%" style="position: absolute; top: 0; right: 0;" alt="Logo du Projet"/>

	# <code>❯ REMPLACEZ-MOI</code>

	<em>Simplifier la Complexité, Révéler des Insights</em>

	<!-- BADGES -->
	<!-- Repository local, pas encore de badges de métadonnées. -->

	<em>Construit avec les outils et technologies :</em>
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
	
</div>
</div>
<br clear="right">

---

## Table des Matières

- [Table des Matières](#table-des-matières)
- [Aperçu](#aperçu)
- [Fonctionnalités](#fonctionnalités)
- [Structure du Projet](#structure-du-projet)
	- [Index du Projet](#index-du-projet)
- [Feuille de Route](#feuille-de-route)
- [Contribuer](#contribuer)
- [Licence](#licence)
- [Remerciements](#remerciements)

---

## Aperçu

Ce projet présente une implémentation d'Analyse en Composantes Principales (ACP) pour réduire la dimensionnalité des données. Il permet de traiter un dataset tel que MNIST, d'en extraire les composantes principales et de visualiser l'impact de la réduction sur les données.

---

## Fonctionnalités

|      | Composant         | Détails                                                                                                        |
| :--- | :---------------- | :------------------------------------------------------------------------------------------------------------- |
| ⚙️   | **Architecture**  | <ul><li>Utilise Python pour le traitement de données</li><li>Gestion simple des fichiers</li></ul>             |
| 🔩   | **Qualité du Code**| <ul><li>Respecte les normes PEP8</li><li>Structure modulaire</li></ul>                                         |
| 📄   | **Documentation**  | <ul><li>Documentation de base fournie</li><li>Possibilité d’amélioration pour une meilleure prise en main</li></ul> |
| 🔌   | **Intégrations**   | <ul><li>Bibliothèques standard de Python</li><li>Aucune dépendance externe hormis les fichiers de dataset</li></ul> |

---

## Structure du Projet

```sh
└── /
		├── 0-main.py
		├── 0-pca.py
		├── 1-main.py
		├── 1-pca.py
		├── README.md
		├── mnist2500_X.txt
		└── mnist2500_labels.txt
```

### Index du Projet

<details open>
	<summary><b><code>/</code></b></summary>
	<!-- __root__ Sous-module -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class="directory-path" style="padding: 8px 0; color: #666;">
				<code><b>⦿ __root__</b></code>
			<table style="width: 100%; border-collapse: collapse;">
				<thead>
					<tr style="background-color: #f8f9fa;">
						<th style="width: 30%; text-align: left; padding: 8px;">Nom du Fichier</th>
						<th style="text-align: left; padding: 8px;">Résumé</th>
					</tr>
				</thead>
				<tr style="border-bottom: 1px solid #eee;">
					<td style="padding: 8px;"><b><a href='/1-main.py'>1-main.py</a></b></td>
					<td style="padding: 8px;">Réduit la dimensionnalité du dataset MNIST à 50 composantes grâce à l’ACP.<br>
					Charge les données, applique l’ACP et affiche les formes des données originales et transformées.</td>
				</tr>
				<tr style="border-bottom: 1px solid #eee;">
					<td style="padding: 8px;"><b><a href='/mnist2500_X.txt'>mnist2500_X.txt</a></b></td>
					<td style="padding: 8px;">Contient les données du dataset (valeurs en notation exponentielle).<br>
					À utiliser avec les autres fichiers pour la réduction dimensionnelle et l’analyse des données.</td>
				</tr>
				<tr style="border-bottom: 1px solid #eee;">
					<td style="padding: 8px;"><b><a href='/0-pca.py'>0-pca.py</a></b></td>
					<td style="padding: 8px;">Implémente l’ACP de manière à extraire les composantes principales en conservant l’essentiel de la variance.<br>
					Prépare les données pour une analyse efficace.</td>
				</tr>
				<tr style="border-bottom: 1px solid #eee;">
					<td style="padding: 8px;"><b><a href='/mnist2500_labels.txt'>mnist2500_labels.txt</a></b></td>
					<td style="padding: 8px;">Contient 2500 étiquettes numériques correspondant aux images (de 0 à 9 en notation scientifique).<br>
					Essentiel pour les tâches de classification supervisée.</td>
				</tr>
				<tr style="border-bottom: 1px solid #eee;">
					<td style="padding: 8px;"><b><a href='/1-pca.py'>1-pca.py</a></b></td>
					<td style="padding: 8px;">Utilise la décomposition en valeurs singulières (SVD) pour réaliser l’ACP.<br>
					Réduit la dimensionnalité tout en conservant les informations essentielles pour la visualisation et l’analyse.</td>
				</tr>
				<tr>
					<td style="padding: 8px;"><b><a href='/0-main.py'>0-main.py</a></b></td>
					<td style="padding: 8px;">Démonstration de l’ACP sur des données synthétiques.<br>
					Applique l’ACP pour projeter, reconstruire et évaluer l’exactitude de la réduction dimensionnelle.</td>
				</tr>
			</table>
		</div>
		</blockquote>
	</details>
</details>

---

## Feuille de Route

- [X] **Tâche 1** : <strike>Implémenter la fonctionnalité de base.</strike>
- [ ] **Tâche 2** : Développer une nouvelle fonctionnalité.
- [ ] **Tâche 3** : Améliorer l’interface utilisateur et la documentation.

---

## Contribuer

- **💬 [Participer aux Discussions](https://LOCAL///discussions)** : Partagez vos idées, vos retours ou posez des questions.
- **🐛 [Signaler des Problèmes](https://LOCAL///issues)** : Soumettez des bugs ou faites des demandes de fonctionnalités.
- **💡 [Proposer des Pull Requests](https://LOCAL///blob/main/CONTRIBUTING.md)** : Examinez les PR ouvertes et proposez les vôtres.

<details closed>
	<summary>Directives pour Contribuer</summary>

	1. **Forker le Dépôt** : Commencez par forker le projet sur votre compte LOCAL.
	2. **Cloner Localement** : Clonez le dépôt forké sur votre machine.
		 ```sh
		 git clone <URL_DU_FORK>
		 ```
	3. **Créer une Nouvelle Branche** : Travaillez toujours sur une branche dédiée, avec un nom descriptif.
		 ```sh
		 git checkout -b nouvelle-fonctionnalité
		 ```
	4. **Développer vos Modifications** : Codez, testez, et vérifiez vos changements localement.
	5. **Commiter vos Changements** : Rédigez un message de commit clair expliquant vos modifications.
		 ```sh
		 git commit -m "Ajout de la nouvelle fonctionnalité X."
		 ```
	6. **Pousser sur LOCAL** : Envoyez vos modifications sur votre dépôt forké.
		 ```sh
		 git push origin nouvelle-fonctionnalité
		 ```
	7. **Ouvrir une Pull Request** : Soumettez une PR sur le dépôt original en expliquant clairement vos changements.
	8. **Revue et Fusion** : Une fois approuvée, votre PR sera fusionnée dans la branche principale.
</details>

<details closed>
	<summary>Graphique des Contributeurs</summary>
	<br>
	<p align="left">
		<a href="https://LOCAL///graphs/contributors">
			<img src="https://contrib.rocks/image?repo=/">
		</a>
	</p>
</details>

---

## Remerciements

- Merci aux contributeurs, aux sources d’inspiration et aux références qui ont rendu ce projet possible.

<div align="right">
	[![][back-to-top]](#top)
</div>

[back-to-top]: https://img.shields.io/badge/-RETOUR_EN_HAUT-151515?style=flat-square
