import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ast import literal_eval
import os


#------------------------------------------------------------
PATH_PROCESSED = './proccessed/'
os.makedirs(PATH_PROCESSED, exist_ok=True)

def load_jsonl_to_dataframe(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Load each line as JSON
            json_data = json.loads(line)
            data.append(json_data)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    return df

def process_list_col_category(col, name_col):
    mp = {}
    cnt = 0
    for tmp_ls in col:
        if isinstance(tmp_ls, list):
            for item in tmp_ls:
                if item not in mp and item not in ['\\N', '', 'None', 'See more company credits at IMDbPro'] :
                    mp[item] = cnt
                    cnt += 1
    ls_mp = [{"id": value, "name": key} for key, value in mp.items()]
    with open(PATH_PROCESSED + name_col + '.json', 'w', encoding='utf-8') as json_file:
        json.dump(ls_mp, json_file, ensure_ascii=False, indent=2)
    return mp, ls_mp

def mapping_category(mp):
    def mapping(row):
        ls = []
        for item in row:
            if item not in ['\\N', '', 'None', 'See more company credits at IMDbPro']:
                ls.append(mp[item])
        return ls
    return mapping

#------------------------------------------------------------

jsonl_file_path = 'final.jsonl'

# Load dữ liệu vào DataFrame
df_raw = load_jsonl_to_dataframe(jsonl_file_path)
df_raw['Director'] = df_raw['Director'].apply(lambda x: [] if x is np.nan else x)
df_raw['Writers'] = df_raw['Writers'].apply(lambda x: [] if x is np.nan else x)
df_raw['Stars'] = df_raw['Stars'].apply(lambda x: [] if x is np.nan else x)
df_raw['country_of_origin'] = df_raw['country_of_origin'].apply(lambda x: [] if x is np.nan else x)
df_raw['production_companies'] = df_raw['production_companies'].apply(lambda x: [] if x is np.nan else x)
df_raw['languages'] = df_raw['languages'].apply(lambda x: [] if x is np.nan else x)
df_raw['genres'] = df_raw['genres'].apply(lambda x: [] if x is np.nan else x)
df_raw['revenue'] = df_raw['revenue'].fillna(0).astype(float)
df_raw['budget'] = df_raw['budget'].fillna(0).astype(float)
df_raw['runtimeMinutes'] = df_raw['runtimeMinutes'].fillna(0)\
            .apply(lambda x: 0 if x == '\\N' else x).astype(float)
#-------------director---------------------
director_col = df_raw['Director']
mp_director, ls_director = process_list_col_category(director_col, 'director')


#--------------writer---------------------
writer_col = df_raw['Writers']
mp_writer, ls_writer = process_list_col_category(writer_col, 'writer')


#----------------star----------------------
star_col = df_raw['Stars']
mp_star, ls_star = process_list_col_category(star_col, 'star')

#-------------------country--------------------
country_col = df_raw['country_of_origin']
mp_country, ls_country = process_list_col_category(country_col, 'country')

#---------------production company----------
company_col = df_raw['production_companies']
mp_company, ls_company = process_list_col_category(company_col, 'company')

#-----------------language-----------------
language_col = df_raw['languages']
mp_language, ls_language = process_list_col_category(language_col, 'language')

#-------------------genre---------------------
genre_col = df_raw['genres']
mp_genre, ls_genre = process_list_col_category(genre_col, 'genre')

#----------------process main data---------------
df_processed = df_raw.copy()
df_processed['Director'] = df_processed['Director'].apply(mapping_category(mp_director))
df_processed['Writers'] = df_processed['Writers'].apply(mapping_category(mp_writer))
df_processed['Stars'] = df_processed['Stars'].apply(mapping_category(mp_star))
df_processed['country_of_origin'] = df_processed['country_of_origin'].apply(mapping_category(mp_country))
df_processed['production_companies'] = df_processed['production_companies'].apply(mapping_category(mp_company))
df_processed['languages'] = df_processed['languages'].apply(mapping_category(mp_language))
df_processed['genres'] = df_processed['genres'].apply(mapping_category(mp_genre))


df_movie_director = df_processed[['imdb_id', 'Director']].explode('Director')
df_movie_writer = df_processed[['imdb_id', 'Writers']].explode('Writers')
df_movie_star = df_processed[['imdb_id', 'Stars']].explode('Stars')
df_movie_country = df_processed[['imdb_id', 'country_of_origin']].explode('country_of_origin')
df_movie_company = df_processed[['imdb_id', 'production_companies']].explode('production_companies')
df_movie_language = df_processed[['imdb_id', 'languages']].explode('languages')
df_movie_genre = df_processed[['imdb_id', 'genres']].explode('genres')

#-------------------save processed data-----------------------------


df_processed.to_csv(PATH_PROCESSED + 'movie.csv')
df_movie_director.to_csv(PATH_PROCESSED + 'movie_director.csv')
df_movie_writer.to_csv(PATH_PROCESSED + 'movie_writer.csv')
df_movie_star.to_csv(PATH_PROCESSED + 'movie_star.csv')
df_movie_country.to_csv(PATH_PROCESSED + 'movie_country.csv')
df_movie_company.to_csv(PATH_PROCESSED + 'movie_company.csv')
df_movie_language.to_csv(PATH_PROCESSED + 'movie_language.csv')
df_movie_genre.to_csv(PATH_PROCESSED + 'movie_genre.csv')