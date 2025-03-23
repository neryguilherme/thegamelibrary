import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import streamlit as st

# Função para filtrar os dados
def filter_data(data, min_price, max_price, tag, category, language):
    data = data[
        (data["Price"] >= min_price) &
        (data["Price"] <= max_price) &
        (data["Tags"].str.contains(tag)) &
        (data["Categories"].str.contains(category)) &
        (data["Supported languages"].str.contains(language))
    ]
    st.write("Shape dos dados filtrados:", data.shape)
    return data

# Função para processar os gêneros
def process_genres(genres_str):
    if not isinstance(genres_str, str):
        return "Outros"
    
    genres = set(g.strip() for g in genres_str.split(','))
    valid_genres = {"Indie", "Action", "Casual"}
    
    filtered_genres = sorted(valid_genres.intersection(genres))
    if len(filtered_genres) < len(genres):
        filtered_genres.append("Outros")
    
    return ", ".join(filtered_genres)

# Função para processar os dados
def process_data(dataset):
    x_column = dataset.copy()
    
    x_column['Genres'] = x_column['Genres'].apply(process_genres)
    label_encoders2 = {}

    # Carrega os label encoders salvos
    for filename in os.listdir('label_encoders'):
        if filename.endswith('.pkl'):
            key = filename[:-4]  # Remove '.pkl'
            filepath = os.path.join('label_encoders', filename)
            with open(filepath, 'rb') as f:
                label_encoders2[key] = pickle.load(f)
    
    # Aplica os transformadores nas colunas
    for column_name, encoder3 in label_encoders2.items():
        if column_name in x_column.columns:
            x_column[column_name] = encoder3.transform(x_column[column_name])
    
    # Armazena os gêneros e os remove do dataframe
    k = x_column['Genres']
    x_column = x_column.drop(columns=['Genres'])

    # Padroniza as colunas numéricas
    scaler = StandardScaler()
    numerical_cols = x_column.select_dtypes(include=['number']).columns
    x_column[numerical_cols] = scaler.fit_transform(x_column[numerical_cols])

    # Codifica os gêneros
    le = LabelEncoder()
    k = le.fit_transform(k)

    return x_column, k, label_encoders2

# Função para realizar a predição usando o modelo XGBoost
def run_xgb(X, y, label_encoders):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    path = os.getcwd()
    xgb_model_path = os.path.join(path, 'xgb_model.pkl')
    xgb_model = pickle.load(open(xgb_model_path, 'rb'))
    
    prediction_encoded = xgb_model.predict(X_test)
    
    prediction = label_encoders['Genres'].inverse_transform(prediction_encoded)
    
    return prediction

# Função principal de predição
def prediction(min_price, max_price, tag, category, language):
    path = os.getcwd()
    file_path = os.path.join(path, 'games_preprocessed.parquet')
    
    # Carrega o dataset pré-processado
    database = pd.read_parquet(file_path)
    
    # Filtra os dados com base nos parâmetros do usuário
    data_filtered = filter_data(database, min_price, max_price, tag, category, language)
    
    # Processa os dados e prepara para predição
    X, y, label_enc = process_data(data_filtered)
    
    # Executa a predição com XGBoost
    predicted = run_xgb(X, y, label_enc)
    
    return predicted

# Integração com Streamlit
st.title("Predição de Gêneros de Jogos")
st.markdown("Ajuste os parâmetros para filtrar os dados e preveja os gêneros dos jogos.")

# Carrega o dataset para extrair os valores dos parâmetros
path = os.getcwd()
file_path = os.path.join(path, 'games_preprocessed.parquet')
database = pd.read_parquet(file_path)

# Definindo os limites de preço com base nos dados
price_min = float(database['Price'].min())
price_max = float(database['Price'].max())

# Extraindo valores únicos para Tags, Categories e Supported languages
def extrair_valores_unicos(coluna):
    # Considerando que os valores podem estar separados por vírgula
    valores = database[coluna].dropna().apply(lambda x: [v.strip() for v in x.split(',')])
    # Flatten e removendo duplicatas
    lista = set([item for sublist in valores for item in sublist])
    return sorted(lista)

unique_tags = extrair_valores_unicos("Tags")
unique_categories = extrair_valores_unicos("Categories")
unique_languages = extrair_valores_unicos("Supported languages")

# Parâmetros de entrada via sidebar
st.sidebar.header("Parâmetros de Entrada")

min_price_input = st.sidebar.number_input("Preço Mínimo", 
                                            value=price_min, 
                                            min_value=price_min, 
                                            max_value=price_max, 
                                            step=1.0)
max_price_input = st.sidebar.number_input("Preço Máximo", 
                                            value=price_max, 
                                            min_value=price_min, 
                                            max_value=price_max, 
                                            step=1.0)

tag_input = st.sidebar.selectbox("Tag", [""] + unique_tags)
category_input = st.sidebar.selectbox("Categoria", [""] + unique_categories)
language_input = st.sidebar.selectbox("Idioma Suportado", [""] + unique_languages)

# Botão para executar a predição
if st.sidebar.button("Prever"):
    with st.spinner("Realizando predição..."):
        predicted = prediction(min_price_input, max_price_input, tag_input, category_input, language_input)
        st.success("Predição realizada!")
        st.write("Gêneros previstos:")
        st.write(predicted)
