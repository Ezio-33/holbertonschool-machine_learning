-- Crée un trigger qui remet valid_email à 0 uniquement quand l'email a été modifié
DELIMITER $$
CREATE TRIGGER reset_valid_email
    BEFORE UPDATE ON users
    FOR EACH ROW
BEGIN
    IF NEW.email != OLD.email THEN
        SET NEW.valid_email = 0;
    END IF;
END$$
DELIMITER ;
