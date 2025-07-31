-- Liste toutes les séries avec leurs genres (même celles sans genre)
-- Format: titre - genre_id, affiche NULL si pas de genre
SELECT tv_shows.title, tv_show_genres.genre_id
FROM tv_shows
LEFT JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id
ORDER BY tv_shows.title, tv_show_genres.genre_id;
