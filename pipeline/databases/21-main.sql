-- Test et comparaison des divisions
SELECT 'Division normale:' as test;
SELECT a, b, (a / b) as division_normale FROM numbers;

SELECT 'Division sécurisée:' as test;  
SELECT a, b, SafeDiv(a, b) as division_securisee FROM numbers;
