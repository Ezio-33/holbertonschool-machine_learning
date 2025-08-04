-- Crée une procédure stockée qui calcule et stocke la moyenne des scores d'un utilisateur
DELIMITER $$
CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id INT
)
BEGIN
    DECLARE avg_score FLOAT;
    
    -- Calcule la moyenne des scores pour cet utilisateur
    SELECT AVG(score) INTO avg_score 
    FROM corrections 
    WHERE corrections.user_id = user_id;
    
    -- Met à jour le champ average_score dans la table users
    UPDATE users 
    SET average_score = IFNULL(avg_score, 0) 
    WHERE id = user_id;
END$$
DELIMITER ;
