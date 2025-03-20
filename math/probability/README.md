<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="readmeai/assets/logos/purple.svg" width="30%" style="position: relative; top: 0; right: 0;" alt="Logo du projet"/>

# PROBABILITÉ

<em></em>

<!-- BADGES -->
<!-- dépôt local, pas de badges de métadonnées. -->

<em>Construit avec les outils et technologies :</em>

<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">

</div>
<br>

---

## Table des matières

- [Table des matières](#table-des-mati%C3%A8res)
- [Aperçu](#aper%C3%A7u)
- [Fonctionnalités](#fonctionnalit%C3%A9s)
- [Structure du projet](#structure-du-projet)
  - [Index du projet](#index-du-projet)
- [Prise en main](#prise-en-main)
  - [Prérequis](#pr%C3%A9requis)
  - [Installation](#installation)
  - [Utilisation](#utilisation)
  - [Tests](#tests)
- [Feuille de route](#feuille-de-route)
- [Contribution](#contribution)
- [Licence](#licence)
- [Remerciements](#remerciements)

---

## Aperçu

---

## Fonctionnalités

|     | Composant | Détails |
| :-- | :-------- | :------ |

---

## Structure du projet

```sh
└── probability/
	├── 0-main.py
	├── 1-main.py
	├── 10-main.py
	├── 11-main.py
	├── 12-main.py
	├── 2-main.py
	├── 3-main.py
	├── 4-main.py
	├── 5-main.py
	├── 6-main.py
	├── 7-main.py
	├── 8-main.py
	├── 9-main.py
	├── README.md
	├── __pycache__
	│   ├── binomial.cpython-310.pyc
	│   ├── exponential.cpython-310.pyc
	│   ├── normal.cpython-310.pyc
	│   └── poisson.cpython-310.pyc
	├── binomial.py
	├── exponential.py
	├── normal.py
	└── poisson.py
```

### Index du projet

<details open>
	<summary><b><code>PROBABILITÉ/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>Nom du fichier</th>
					<th style='text-align: left; padding: 8px;'>Résumé</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/binomial.py'>binomial.py</a></b></td>
					<td style='padding: 8px;'>Implémente un modèle de distribution binomiale pour calculer la probabilité de succès dans des essais indépendants<br>- Fournit des méthodes pour calculer la fonction de masse de probabilité (PMF) et la fonction de répartition cumulative (CDF)<br>- Prend en charge l'initialisation avec des paramètres spécifiés ou l'estimation à partir d'échantillons de données.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/poisson.py'>poisson.py</a></b></td>
					<td style='padding: 8px;'>Implémente une classe de distribution de Poisson pour modéliser la probabilité d'événements sur des intervalles fixes<br>- Prend en charge l'estimation des paramètres à partir des données d'entrée ou par spécification directe<br>- Calcule la fonction de masse de probabilité (PMF) et la fonction de répartition cumulative (CDF) pour l'analyse statistique<br>- Utile pour la modélisation probabiliste dans des scénarios avec des taux moyens d'événements connus.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/11-main.py'>11-main.py</a></b></td>
					<td style='padding: 8px;'>Démontre l'utilisation de la classe Binomial en l'instanciant avec des données d'exemple et des paramètres explicites<br>- Illustre le calcul de la fonction de masse de probabilité (PMF) à une valeur spécifique, démontrant ainsi la capacité de la bibliothèque à modéliser et analyser efficacement les distributions binomiales.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/exponential.py'>exponential.py</a></b></td>
					<td style='padding: 8px;'>Modélise une distribution exponentielle pour analyser les intervalles de temps entre des événements survenant à un taux constant<br>- Estime les paramètres à partir des données ou d'une entrée directe<br>- Calcule la fonction de densité de probabilité (PDF) et la fonction de répartition cumulative (CDF) pour l'analyse du timing des événements<br>- Fournit des outils statistiques pour modéliser les probabilités d'événements rares dans diverses applications.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/5-main.py'>5-main.py</a></b></td>
					<td style='padding: 8px;'>Démontre l'utilisation de la classe Exponential en créant des instances à partir de données d'exemple et d'un paramètre lambda spécifié<br>- Calcule et compare les valeurs de la fonction de répartition cumulative (CDF) à un point donné pour illustrer la fonctionnalité.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/8-main.py'>8-main.py</a></b></td>
					<td style='padding: 8px;'>Démontre la fonctionnalité de l'implémentation de la distribution Normale en calculant la densité de probabilité (PDF) à une valeur spécifique (90)<br>- Compare les résultats obtenus avec des données réelles et des paramètres théoriques, montrant comment la classe Normal gère différentes entrées et valide son exactitude<br>- Sert d'exemple pour aider les utilisateurs à comprendre et appliquer une logique similaire dans leurs flux de travail.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/4-main.py'>4-main.py</a></b></td>
					<td style='padding: 8px;'>Démontre l'utilisation de la classe Exponential en générant des données exponentielles<br>- Crée des instances à partir de données et avec un paramètre lambda, puis calcule les valeurs de la fonction de densité de probabilité (PDF)<br>- Fournit des exemples pratiques pour comprendre et appliquer les concepts de distribution exponentielle dans l'analyse statistique.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/6-main.py'>6-main.py</a></b></td>
					<td style='padding: 8px;'>Démontre l'utilisation de la classe Normal en créant des instances soit à partir de données d'exemple, soit à partir de paramètres spécifiés<br>- Calcule et affiche des mesures statistiques telles que la moyenne et l'écart type pour illustrer les propriétés de la distribution<br>- Fournit des exemples clairs pour travailler avec les distributions normales dans le cadre du projet.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/1-main.py'>1-main.py</a></b></td>
					<td style='padding: 8px;'>Démontre la fonctionnalité de la classe Poisson en utilisant des données d'exemple et une valeur lambda spécifiée<br>- Calcule la fonction de masse de probabilité (PMF) pour illustrer le comportement de la distribution dans des scénarios pratiques.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/0-main.py'>0-main.py</a></b></td>
					<td style='padding: 8px;'>Démontre l'utilisation de la classe Poisson en créant deux instances : l'une à partir de données générées pour estimer λ et l'autre avec une valeur λ spécifiée<br>- Illustre la méthode d'accès au paramètre estimé, mettant en avant la flexibilité de la classe pour différents scénarios.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/3-main.py'>3-main.py</a></b></td>
					<td style='padding: 8px;'>Démontre la création et l'utilisation d'objets Exponential en les instanciant avec des données générées et des paramètres spécifiés<br>- Met en évidence l'utilisation de ces objets dans le code pour modéliser des distributions exponentielles, illustrant ainsi leur intégration dans la fonctionnalité du projet.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/2-main.py'>2-main.py</a></b></td>
					<td style='padding: 8px;'>Démontre la fonctionnalité de la distribution de Poisson en créant des instances à partir de données d'exemple et de valeurs théoriques de lambda<br>- Compare les calculs de la fonction de répartition cumulative (CDF) pour valider l'exactitude de l'implémentation<br>- Offre des exemples clairs d'utilisation pratique dans l'analyse statistique.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/normal.py'>normal.py</a></b></td>
					<td style='padding: 8px;'>Modélise des distributions normales en calculant des mesures statistiques clés telles que les scores z, les valeurs x, la fonction de densité de probabilité (PDF), la fonction de répartition cumulative (CDF) et des approximations de la fonction d'erreur<br>- Permet une analyse probabiliste et une modélisation statistique pour diverses applications nécessitant des calculs sur la distribution normale.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/7-main.py'>7-main.py</a></b></td>
					<td style='padding: 8px;'>Démontre l'utilisation de la classe Normal en créant des instances à partir de données d'exemple et de paramètres explicites<br>- Calcule et affiche des scores z ainsi que des valeurs x pour des entrées données, illustrant ainsi la fonctionnalité de la distribution normale selon différentes méthodes d'initialisation.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/10-main.py'>10-main.py</a></b></td>
					<td style='padding: 8px;'>Démontre la fonctionnalité de la classe Binomial en créant des instances à partir de données d'exemple et de paramètres spécifiés<br>- Met en évidence la manière dont n et p sont calculés pour les distributions binomiales.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/9-main.py'>9-main.py</a></b></td>
					<td style='padding: 8px;'>Démontre l'implémentation de la distribution normale en calculant la fonction de répartition cumulative (CDF) pour des distributions dérivées de données et définies par des paramètres<br>- Illustre l'utilisation pratique de la classe Normal pour évaluer les probabilités à des points spécifiques, montrant son applicabilité dans des scénarios d'analyse statistique.</td>
				</tr>
				<tr>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/probability/blob/master/12-main.py'>12-main.py</a></b></td>
					<td style='padding: 8px;'>Démontre l'implémentation de la classe Binomial en ajustant des données d'exemple ou en spécifiant des paramètres<br>- Calcule et affiche la fonction de répartition cumulative à 30, illustrant ainsi son utilisation pour des distributions binomiales empiriques et théoriques.</td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## Prise en main

### Prérequis

Ce projet requiert les dépendances suivantes :

- **Langage de programmation :** Python

### Installation

Construisez probability depuis la source et installez les dépendances :

1. **Cloner le dépôt :**

   ```sh
   ❯ git clone ../probability
   ```

2. **Naviguer jusqu'au répertoire du projet :**

   ```sh
   ❯ cd probability
   ```

3. **Installer les dépendances :**

echo 'INSERT-INSTALL-COMMAND-HERE'

### Utilisation

Exécutez le projet avec :

echo 'INSERT-RUN-COMMAND-HERE'

### Tests

Probability utilise le framework de test {**test_framework**}. Exécutez la suite de tests avec :

echo 'INSERT-TEST-COMMAND-HERE'

---

## Feuille de route

- [x] **`Task 1`** : <strike>Implémenter la fonctionnalité une.</strike>
- [ ] **`Task 2`** : Implémenter la fonctionnalité deux.
- [ ] **`Task 3`** : Implémenter la fonctionnalité trois.

---

## Contribution

- **💬 [Rejoignez les discussions](https://LOCAL/math/probability/discussions)** : Partagez vos idées, vos retours ou posez des questions.
- **🐛 [Signalez des problèmes](https://LOCAL/math/probability/issues)** : Soumettez des bugs ou proposez des demandes d'amélioration pour le projet `probability`.
- **💡 [Soumettez des demandes de tirage](https://LOCAL/math/probability/blob/main/CONTRIBUTING.md)** : Consultez les PR en attente et soumettez la vôtre.

<details closed>
<summary>Directives pour les contributeurs</summary>

1. **Forkez le dépôt** : Commencez par forker le dépôt du projet vers votre compte LOCAL.
2. **Clonez localement** : Clonez le dépôt forqué sur votre machine locale avec un client git.
   ```sh
   git clone /root/Projets_holberton/holbertonschool-machine_learning/math/probability
   ```
3. **Créez une nouvelle branche** : Travaillez toujours sur une nouvelle branche, en lui donnant un nom descriptif.
   ```sh
   git checkout -b nouvelle-fonctionnalite-x
   ```
4. **Apportez vos modifications** : Développez et testez vos modifications en local.
5. **Validez vos modifications** : Effectuez un commit avec un message clair décrivant vos mises à jour.
   ```sh
   git commit -m 'Implémentation de la fonctionnalité x.'
   ```
6. **Poussez vers LOCAL** : Poussez vos modifications sur votre dépôt forqué.
   ```sh
   git push origin nouvelle-fonctionnalite-x
   ```
7. **Soumettez une demande de tirage** : Créez une PR contre le dépôt original en expliquant clairement les changements et leurs motivations.
8. **Revue** : Une fois votre PR revue et approuvée, elle sera fusionnée dans la branche principale. Félicitations pour votre contribution !
</details>

---

<div align="right">

[![][back-to-top]](#top)

</div>

[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square
