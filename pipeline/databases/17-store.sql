-- Crée un trigger qui diminue la quantité d'un article après ajout d'une nouvelle commande
DELIMITER $$
CREATE TRIGGER decrease_quantity
    AFTER INSERT ON orders
    FOR EACH ROW
BEGIN
    UPDATE items 
    SET quantity = quantity - NEW.number 
    WHERE name = NEW.item_name;
END$$
DELIMITER ;
