import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path)

st.title("Nuvem de Palavras - Tags e Gêneros dos Jogos")
st.write("Visualização das tags e gêneros mais frequentes utilizando uma nuvem de palavras.")

file_path = "games_cleaned.parquet"

if os.path.exists(file_path):
    try:
        
        data = load_data(file_path)
        
        if "Tags" in data.columns and "Genres" in data.columns:
            data = data.dropna(subset=["Tags", "Genres"])
            
            # Juntar todas as tags e gêneros em uma única string
            all_tags = " ".join(data["Tags"].astype(str))
            all_genres = " ".join(data["Genres"].astype(str))
            
            # Criar uma única string com todas as tags e gêneros
            text_for_wordcloud = all_tags + " " + all_genres
            
            # Gerar a nuvem de palavras
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_for_wordcloud)
            
            # Exibir a nuvem de palavras
            st.subheader("Nuvem de Palavras - Tags e Gêneros")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")  # Remover os eixos
            st.pyplot(fig)
        else:
            st.error("O arquivo deve conter as colunas 'Tags' e 'Genres'.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")
else:
    st.error(f"O arquivo '{file_path}' não foi encontrado.")
