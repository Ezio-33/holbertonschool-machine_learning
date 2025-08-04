-- Crée une fonction SafeDiv qui divise le premier par le second nombre ou retourne 0 si le second nombre est égal à 0
DELIMITER $$
CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS FLOAT
READS SQL DATA
DETERMINISTIC
BEGIN
    IF b = 0 THEN
        RETURN 0;
    ELSE
        RETURN CAST(a AS FLOAT) / CAST(b AS FLOAT);
    END IF;
END$$
DELIMITER ;
