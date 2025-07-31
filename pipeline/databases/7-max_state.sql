-- Affiche la température maximale de chaque état (ordonné par nom d'état)
SELECT state, MAX(value) AS max_temp FROM temperatures GROUP BY state ORDER BY state;
