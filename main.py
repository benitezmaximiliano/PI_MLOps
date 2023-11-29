from fastapi import FastAPI
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# load data to df from parquet
Tabla_API = pd.read_parquet('gzip/dfAPI.parquet')



@app.get('/PlayTimeGenre')
def PlayTimeGenre(genero: str):
    
    df_genre = Tabla_API[Tabla_API['Genres'].apply(lambda x: genero in x)]

    if df_genre.empty:
        return 'No hay datos para el género especificado.'

    #Find data year
    year_max_playtime = df_genre.groupby('Year')['Playtime_Forever'].sum().idxmax()

    #make result
    return {'Año de lanzamiento con más horas jugadas para Género ' + genero: int(year_max_playtime)}

@app.get('/UserForGenre')
def UserForGenre(genero: str):
    #find with genres 
    df_genre = Tabla_API[Tabla_API['Genres'].apply(lambda x: genero in x)]
    #check is empty
    if df_genre.empty:
        return 'No hay datos para el género especificado.'

    #find user 
    user_max_hours = df_genre.groupby('User_Id')['Playtime_Forever'].sum().idxmax()

    hours_per_year = df_genre.groupby('Posted')['Playtime_Forever'].sum().reset_index().to_dict('records')

    #make result
    return ({
        'Usuario con más horas jugadas para Género ' + genero: user_max_hours,
        'Horas jugadas': hours_per_year
    })

@app.get('/UsersRecommend')
def UsersRecommend(year: int):
    
    df_year = Tabla_API[(Tabla_API['Year'] == year) & (Tabla_API['Recommend'] == True)]

    # sum recommend
    df_count = df_year.groupby('Title')['Recommend'].count()

    #top 3
    top_games = df_count.nlargest(3)

    #create list with result
    return [{'Puesto ' + str(i+1): game} for i, game in enumerate(top_games.index)]

@app.get('/UsersWorstDeveloper')
def UsersWorstDeveloper(year: int):
    
    df_year = Tabla_API[(Tabla_API['Year'] == year) & (Tabla_API['Recommend'] == False)]

    #find developer with negative recommend
    df_count = df_year.groupby('Developer')['Recommend'].count()

    #top 3
    worst_developers = df_count.nlargest(3)

    return [{'Puesto ' + str(i+1): developer} for i, developer in enumerate(worst_developers.index)]

@app.get('/sentiment_analysis')
def sentiment_analysis(developer: str):
    
    df_developer = Tabla_API[Tabla_API['Developer'] == developer]

    
    if df_developer.empty:
        return 'No hay datos para el desarrollador especificado.'

    sentiment_counts = df_developer['Sentiment_Score'].value_counts().to_dict()

    return [{developer: {'Negative': sentiment_counts.get(0, 0), 'Neutral': sentiment_counts.get(1, 0), 'Positive': sentiment_counts.get(2, 0)}}]

@app.get('/recomendacion_juego')
def recomendacion_juego(item_id: int):
    #load data
    df = pd.read_csv('csv/gsteam.csv')
    df1 = pd.read_csv('csv/gid.csv')

    #make vector
    tfidv = TfidfVectorizer(min_df=2, max_df=0.7, token_pattern=r'\b[a-zA-Z0-9]\w+\b')
    data_vector = tfidv.fit_transform(df['features'])

    data_vector_df = pd.DataFrame(data_vector.toarray(), index=df['Item_Id'], columns = tfidv.get_feature_names_out())
    vector_similitud_coseno = cosine_similarity(data_vector_df.values)
    cos_sim_df = pd.DataFrame(vector_similitud_coseno, index=data_vector_df.index, columns=data_vector_df.index)
    juego_simil = cos_sim_df.loc[item_id]
    simil_ordenada = juego_simil.sort_values(ascending=False)
    resultado = simil_ordenada.head(6).reset_index()
    result_df = resultado.merge(df1, on='Item_Id',how='left')

    juego_title = df1[df1['Item_Id'] == item_id]['Title'].values[0]

    mensaje = f"Si te gustó el juego {item_id} : {juego_title}, también te pueden gustar:"

    result_dict = {
        'mensaje': mensaje,
        'juegos recomendados': result_df['Title'][1:6].tolist()
    }

    return result_dict
