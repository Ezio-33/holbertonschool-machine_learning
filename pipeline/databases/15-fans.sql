-- Classe les pays d'origine des groupes par nombre total de fans (ordre décroissant)
-- Somme tous les fans par pays d'origine
SELECT origin, SUM(fans) AS nb_fans
FROM metal_bands
GROUP BY origin
ORDER BY nb_fans DESC;
