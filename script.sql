-- script tạo các bảng trong database movie

create schema movie;

CREATE TABLE movie.movie_raw (
    imdb_id text,
    director text,
    writers text,
    stars text,
    summary text,
    release_date text,
    country_of_origin text,
    production_companies text,
    languages text,
    tconst text,
    originalTitle text,
    startYear text,
    runtimeMinutes text,
    genres text,
    revenue numeric,
    budget numeric
);

CREATE TABLE movie.movie (
    imdb_id text,
    director text,  -- list id director
    writers text,   -- list id writers
    stars text,     -- list id stars
    summary text,
    release_date timestamp,
    country_of_origin text, -- list id country
    production_companies text, -- list id company
    languages text,  -- list id languages
    -- tconst text, Trùng với imdb_id
    originalTitle text,
    startYear int,
    runtimeMinutes int,
    genres text,  -- list id genres
    revenue numeric,
    budget numeric
);

create table movie.director (
    id int,
    name text
);

create table movie.writer (
    id int,
    name text
);

create table movie.star (
    id int,
    name text
);

create table movie.country (
    id int,
    name text
);

create table movie.company (
    id int,
    name text
);

create table movie.language (
    id int,
    name text
);

create table movie.genre (
    id int,
    name text
);

create table movie.movie_director (
    unique_id text,
    movie_id text,
    director_id int
);

create table movie.movie_writer (
    unique_id text,
    movie_id text,
    writer_id int
);

create table movie.movie_star (
    unique_id text,
    movie_id text,
    star_id int
);

create table movie.movie_country (
    unique_id text,
    movie_id text,
    country_id int
);

create table movie.movie_company (
    unique_id text,
    movie_id text,
    company_id int
);

create table movie.movie_language (
    unique_id text,
    movie_id text,
    language_id int
);

create table movie.movie_genre (
    unique_id text,
    movie_id text,
    genre_id int
);
