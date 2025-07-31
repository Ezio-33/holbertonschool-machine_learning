-- Liste toutes les séries qui ont au moins un genre lié
-- Format: titre - genre_id, trié par titre puis genre_id
SELECT tv_shows.title, tv_show_genres.genre_id
FROM tv_shows
INNER JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id
ORDER BY tv_shows.title, tv_show_genres.genre_id;
