<div id="top">

<!-- STYLE D'EN-TÊTE : CLASSIQUE -->
<div align="center">

<img src="readmeai/assets/logos/purple.svg" width="30%" style="position: relative; top: 0; right: 0;" alt="Logo du Projet"/>

# ADVANCED_LINEAR_ALGEBRA

<em>Libérez la puissance des matrices : résolvez tout problème d'algèbre linéaire</em>

<!-- BADGES -->
<!-- dépôt local, aucun badge de métadonnées. -->

<em>Construit avec les outils et technologies suivants :</em>

<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">

</div>
<br>

---

## Table des Matières

- [Table des Matières](#table-des-matières)
- [Aperçu](#aperçu)
- [Fonctionnalités](#fonctionnalités)
- [Structure du Projet](#structure-du-projet)
  - [Index du Projet](#index-du-projet)
- [Commencer](#commencer)
  - [Prérequis](#prérequis)
  - [Installation](#installation)
  - [Utilisation](#utilisation)
  - [Tests](#tests)
- [Feuille de Route](#feuille-de-route)
- [Contribution](#contribution)
- [Licence](#licence)
- [Remerciements](#remerciements)

---

## Aperçu

[ Brève description du projet ]

**Pourquoi advanced_linear_algebra ?**

Ce projet offre un ensemble d’outils efficace et complet pour les opérations d’algèbre linéaire, visant à résoudre les difficultés courantes du calcul des déterminants et des inverses.

- **🔹 Calcul du déterminant :** Calculez rapidement les déterminants avec prise en charge intégrée pour diverses tailles de matrices.
- **💡 Calcul de l’adjugée :** Calculez facilement l’adjugée classique d’une matrice carrée.
- **🔄 Calcul de l’inverse :** Calculez l’inverse d’une matrice en utilisant l’adjugée et le déterminant, si elle existe.
- **📝 Opération sur la matrice mineure :** Extraire la matrice mineure en supprimant chaque élément et son mineur correspondant.
- **🔍 Fonction de définitude :** Déterminez le caractère (définit positive, semi-définie positive, etc.) d’une matrice symétrique carrée donnée.

---

## Fonctionnalités

| Composant                                                                                       | Détails                                                                                        |
| :---------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- |
| **Calcul Distribué**                                                                            | • Non conçu pour le calcul distribué, mais peut être parallélisé via joblib ou multiprocessing |
| • Inclut un exemple simple d’utilisation de la bibliothèque dans un environnement multi-thread  |
| • Peut gérer de grandes matrices et systèmes d’équations linéaires avec suffisamment de mémoire |

---

## Structure du Projet

```sh
└── advanced_linear_algebra/
	├── 0-determinant.py
	├── 0-main.py
	├── 1-main.py
	├── 1-minor.py
	├── 2-cofactor.py
	├── 2-main.py
	├── 3-adjugate.py
	├── 3-main.py
	├── 4-inverse.py
	├── 4-main.py
	├── 5-definiteness.py
	├── 5-main.py
	├── README.md
	└── __pycache__
		├── 0-determinant.cpython-310.pyc
		├── 1-minor.cpython-310.pyc
		├── 2-cofactor.cpython-310.pyc
		├── 3-adjugate.cpython-310.pyc
		├── 4-inverse.cpython-310.pyc
		└── 5-definiteness.cpython-310.pyc
```

### Index du Projet

<details open>
	<summary><b><code>ADVANCED_LINEAR_ALGEBRA/</code></b></summary>
	<!-- Sous-module __root__ -->
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
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/advanced_linear_algebra/blob/master/1-minor.py'>1-minor.py</a></b></td>
					<td style='padding: 8px;'>- Calculs matriciels : Le fichier <code>minor.py</code> propose des fonctions pour calculer le déterminant et le mineur d'une matrice.<br>- La fonction <code>determinant</code> calcule le déterminant d'une matrice carrée, tandis que <code>minor</code> extrait la matrice mineure en supprimant chaque élément et son mineur.<br>- Ce code est utile pour les opérations et analyses matricielles en algèbre linéaire.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/advanced_linear_algebra/blob/master/3-adjugate.py'>3-adjugate.py</a></b></td>
					<td style='padding: 8px;'>- Ce module fournit des fonctions pour calculer le déterminant, la matrice des mineurs et l’adjugée d'une matrice carrée.<br>- Il permet de résoudre des problèmes liés à l’algèbre linéaire et aux équations quadratiques.<br>- Les fonctions sont bien documentées et faciles à utiliser, facilitant leur intégration dans d'autres projets.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/advanced_linear_algebra/blob/master/4-inverse.py'>4-inverse.py</a></b></td>
					<td style='padding: 8px;'>- Opérations matricielles : Le code implémente diverses opérations matricielles, notamment le calcul du déterminant, l’adjugée (aussi appelée adjugée classique) et l'inverse.<br>- Les fonctions <code>determinant</code>, <code>adjugate</code> et <code>inverse</code> prennent en entrée une matrice (liste de listes) et réalisent leurs calculs via l’expansion par cofacteurs.<br>- La fonction d'inverse utilise l’adjugée et le déterminant pour calculer l’inverse d'une matrice, si celui-ci existe. Le code suit des normes de lisibilité et de maintenabilité professionnelles.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/advanced_linear_algebra/blob/master/5-main.py'>5-main.py</a></b></td>
					<td style='padding: 8px;'>- Démonstration de la fonctionnalité de définitude : Le fichier <code>5-main.py</code> illustre le comportement de la fonction de définitude sur différentes matrices.<br>- Il importe les bibliothèques nécessaires, définit plusieurs matrices de test, et appelle la fonction <code>definiteness</code> pour évaluer leur caractère.<br>- L'affichage présente les résultats pour la plupart des matrices et lève une exception pour un input non conforme (<code>mat6</code>).</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/advanced_linear_algebra/blob/master/4-main.py'>4-main.py</a></b></td>
					<td style='padding: 8px;'>- Le fichier <code>4-main.py</code> démontre l'inversion de matrices en utilisant la fonction <code>inverse</code> du module <code>4-inverse</code>.<br>- Il présente la fonctionnalité en inversant plusieurs matrices, y compris des cas valides et plusieurs cas d'erreurs, soulignant ainsi les exceptions potentielles.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/advanced_linear_algebra/blob/master/1-main.py'>1-main.py</a></b></td>
					<td style='padding: 8px;'>- Activation de la fonctionnalité de matrice mineure : Le fichier <code>1-main.py</code> sert de point d'entrée pour le projet, important et utilisant la fonction <code>minor</code> du module <code>1-minor</code> pour traiter diverses matrices.<br>- Le code montre l'application de l'opération sur différentes matrices et gère les exceptions pour des matrices non supportées ou vides.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/advanced_linear_algebra/blob/master/0-main.py'>0-main.py</a></b></td>
					<td style='padding: 8px;'>- Calcul des déterminants : Le fichier <code>0-main.py</code> est le point d'entrée pour le calcul des déterminants de diverses matrices.<br>- Il importe et utilise des fonctions d'autres modules pour effectuer les calculs, gérant à la fois les cas réussis et les exceptions, démontrant ainsi un cadre robuste pour les calculs numériques.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/advanced_linear_algebra/blob/master/3-main.py'>3-main.py</a></b></td>
					<td style='padding: 8px;'>- Calcul des déterminants de matrices : Le fichier <code>3-main.py</code> est le point d'entrée pour calculer les déterminants de diverses matrices en utilisant la fonction <code>adjugate</code> du module <code>3-adjugate</code>.<br>- Il teste la fonctionnalité avec des matrices carrées et non carrées, et gère les exceptions pour des inputs invalides.<br>- Le code fournit une base de calcul de déterminants pour un développement ultérieur ou une intégration dans un système plus vaste.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/advanced_linear_algebra/blob/master/2-main.py'>2-main.py</a></b></td>
					<td style='padding: 8px;'>- Le fichier <code>2-main.py</code> sert à tester et démontrer la fonctionnalité du module <code>cofactor</code>, qui calcule le cofacteur d'une matrice donnée.<br>- L’architecture du code se concentre sur les opérations d’algèbre linéaire, avec plusieurs matrices définies pour les tests.<br>- La fonction <code>cofactor</code> est importée du module <code>2-cofactor</code> et utilisée pour calculer les cofacteurs dans divers scénarios.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/advanced_linear_algebra/blob/master/2-cofactor.py'>2-cofactor.py</a></b></td>
					<td style='padding: 8px;'>- Ce module fournit des fonctions pour calculer le déterminant, la matrice des mineurs d'une matrice carrée, ainsi que la matrice des cofacteurs.<br>- Il traite des matrices non nulles et carrées, et retourne les résultats sous forme d’entiers ou de floats.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/advanced_linear_algebra/blob/master/5-definiteness.py'>5-definiteness.py</a></b></td>
					<td style='padding: 8px;'>- Détermine la définitude d'une matrice<br>- Le fichier <code>5-definiteness.py</code> fournit une fonction Python pour déterminer le caractère (définit positive, semi-définie positive, négative semi-définie, négative définie ou indéfinie) d'une matrice symétrique carrée, en utilisant la décomposition en valeurs propres et en vérifiant leur réalité.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/root/Projets_holberton/holbertonschool-machine_learning/math/advanced_linear_algebra/blob/master/0-determinant.py'>0-determinant.py</a></b></td>
					<td style='padding: 8px;'>- Calcule le déterminant d'une matrice carrée : Ce module propose une fonction <code>determinant</code> qui prend en entrée une matrice (liste de listes) et renvoie son déterminant.<br>- La fonction gère les cas particuliers pour les matrices de dimensions 0×0, 1×1, 2×2 et n×n (pour n>2), et lève une exception si la matrice n'est pas carrée ou mal formée.</td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## Commencer

### Prérequis

Ce projet nécessite les dépendances suivantes :

- **Langage de programmation :** Python

### Installation

Construisez advanced_linear_algebra à partir des sources et installez les dépendances :

1. **Cloner le dépôt :**

   ```sh
   ❯ git clone ../advanced_linear_algebra
   ```

2. **Naviguer dans le répertoire du projet :**

   ```sh
   ❯ cd advanced_linear_algebra
   ```

3. **Installer les dépendances :**

   echo 'INSERT-INSTALL-COMMAND-HERE'

### Utilisation

Exécutez le projet avec :

echo 'INSERT-RUN-COMMAND-HERE'

### Tests

advanced_linear_algebra utilise le framework de test {**test_framework**}. Exécutez la suite de tests avec :

echo 'INSERT-TEST-COMMAND-HERE'

---

## Feuille de Route

- [x] **`Tâche 1`** : <strike>Implémentation de la première fonctionnalité.</strike>
- [ ] **`Tâche 2`** : Implémenter la deuxième fonctionnalité.
- [ ] **`Tâche 3`** : Implémenter la troisième fonctionnalité.

---

## Contribution

- **💬 [Participer aux discussions](https://LOCAL/math/advanced_linear_algebra/discussions)** : Partagez vos idées, vos retours ou posez des questions.
- **🐛 [Signaler un problème](https://LOCAL/math/advanced_linear_algebra/issues)** : Signalez les bugs ou proposez des améliorations pour le projet `advanced_linear_algebra`.
- **💡 [Soumettre des Pull Requests](https://LOCAL/math/advanced_linear_algebra/blob/main/CONTRIBUTING.md)** : Consultez les PR ouvertes et soumettez la vôtre.

<details closed>
<summary>Directives de Contribution</summary>

1. **Forker le dépôt** : Commencez par forker le dépôt du projet sur votre compte LOCAL.
2. **Cloner en local** : Clonez le dépôt forker sur votre machine en utilisant un client Git.
   ```sh
   git clone /root/Projets_holberton/holbertonschool-machine_learning/math/advanced_linear_algebra
   ```
3. **Créer une nouvelle branche** : Travaillez toujours sur une branche distincte et donnez-lui un nom descriptif.
   ```sh
   git checkout -b nouvelle-fonctionnalité-x
   ```
4. **Apporter vos modifications** : Développez et testez vos modifications en local.
5. **Valider vos changements** : Effectuez un commit avec un message clair décrivant vos mises à jour.
   ```sh
   git commit -m 'Implémentation de la fonctionnalité x.'
   ```
6. **Pousser vers LOCAL** : Poussez vos changements vers votre dépôt forké.
   ```sh
   git push origin nouvelle-fonctionnalité-x
   ```
7. **Soumettre une Pull Request** : Créez une PR contre le dépôt original en décrivant clairement vos modifications et leurs justifications.
8. **Revue** : Une fois votre PR examinée et approuvée, elle sera fusionnée dans la branche principale. Félicitations pour votre contribution !
</details>

<details closed>
<summary>Graphique des Contributeurs</summary>
<br>
<p align="left">
   <a href="https://LOCAL{/math/advanced_linear_algebra/}graphs/contributors">
	  <img src="https://contrib.rocks/image?repo=math/advanced_linear_algebra">
   </a>
</p>
</details>

<div align="right">

[![][back-to-top]](#top)

</div>

[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square
