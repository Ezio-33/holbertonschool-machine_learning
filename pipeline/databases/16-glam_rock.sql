-- Liste tous les groupes de Glam rock triés par leur longévité
-- Calcule la durée de vie jusqu'en 2020 en utilisant formed et split
SELECT band_name, 
       (IFNULL(split, 2020) - formed) AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC;
