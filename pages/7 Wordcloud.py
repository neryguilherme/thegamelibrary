import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path)

# Função para gerar a nuvem de palavras
def generate_wordcloud(text_data, title):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        max_words=200
    ).generate(text_data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16)
    st.pyplot(fig)

st.title("Análise de Tags e Gêneros de Jogos")
st.write("Visualização das tags e gêneros mais frequentes usando uma nuvem de palavras.")

uploaded_file = st.file_uploader("Envie o arquivo Parquet:", type=["parquet"])
if uploaded_file:
    data = load_data(uploaded_file)
    
    if "tags" in data.columns and "genres" in data.columns:
        # Unificar os textos de tags e gêneros
        all_tags = " ".join(data["tags"].dropna().str.replace(",", " "))
        all_genres = " ".join(data["genres"].dropna().str.replace(",", " "))
        
        # Criar abas para exibir nuvens separadas
        tab_tags, tab_genres = st.tabs(["Tags", "Gêneros"])
        
        with tab_tags:
            st.subheader("Nuvem de Palavras - Tags")
            generate_wordcloud(all_tags, "Tags mais frequentes")
        
        with tab_genres:
            st.subheader("Nuvem de Palavras - Gêneros")
            generate_wordcloud(all_genres, "Gêneros mais frequentes")
    else:
        st.error("O arquivo não contém as colunas 'tags' e 'genres'.")
else:
    st.info("Aguarde o envio de um arquivo Parquet para começar.")
