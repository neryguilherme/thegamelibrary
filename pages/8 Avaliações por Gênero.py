import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path)

st.title("Análise de Avaliações Positivas por Gênero de Jogos")
st.write("Visualização da média de avaliações positivas por gênero utilizando um gráfico de barras.")

# Gêneros para filtragem
allowed_genres = ["Action", "Racing", "Adventure", "RPG", "FPS", "Simulation", "Strategy", "Horror", "Casual", "Management"]

file_path = "games_cleaned.parquet"

if os.path.exists(file_path):
    try:
        data = load_data(file_path)
        
        # Verificar se as colunas necessárias existem
        if "Genres" in data.columns and "Positive" in data.columns:

            # Filtrar colunas
            data = data.dropna(subset=["Genres", "Positive"])
            
            # Explodir a coluna 'Genres' para lidar com múltiplos gêneros
            data["Genres"] = data["Genres"].str.split(",")  
            data_exploded = data.explode("Genres")

            # Filtrar apenas os gêneros desejados
            data_exploded = data_exploded[data_exploded["Genres"].isin(allowed_genres)]

            # Calcular a média de avaliações positivas por gênero
            avg_positive_by_genre = data_exploded.groupby("Genres")["Positive"].mean().reset_index()

            st.subheader("Média de Avaliações Positivas por Gênero")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=avg_positive_by_genre, x="Genres", y="Positive", palette="viridis", ax=ax)
            ax.set_title("Média de Avaliações Positivas por Gênero", fontsize=16)
            ax.set_xlabel("Gêneros", fontsize=12)
            ax.set_ylabel("Média de Avaliações Positivas", fontsize=12)
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)
        else:
            st.error("O arquivo deve conter as colunas 'Genres' e 'Positive'.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")
else:
    st.error(f"O arquivo '{file_path}' não foi encontrado.")