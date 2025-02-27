import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os

# Caminho do arquivo .parquet (modifique conforme necessário)
FILE_PATH = "games.parquet"

# Função para carregar os dados com cache
@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path)

# Função para extrair valores únicos da coluna 'Genres'
def get_unique_genres(df, column="Genres"):
    all_genres = set()  # Usamos um conjunto para evitar repetições
    df[column].dropna().astype(str).apply(lambda x: all_genres.update(x.split(',')))  # Separar por vírgula
    return {genre.strip() for genre in all_genres}  # Remover espaços extras

# Função principal da aplicação
def main():
    st.title("Análise de Gêneros - Dataset Parquet")

    # Verificar se o arquivo existe
    if not os.path.exists(FILE_PATH):
        st.error(f"O arquivo '{FILE_PATH}' não foi encontrado. Verifique o caminho e tente novamente.")
        return

    # Carregar os dados
    df = load_data(FILE_PATH)

    # Verificar se a coluna 'Genres' existe no dataset
    if 'Genres' in df.columns:
        # Obter gêneros únicos
        unique_genres = get_unique_genres(df, "Genres")
        st.subheader(f"Número total de gêneros únicos: {len(unique_genres)}")

        # Contar os gêneros mais frequentes (considerando separação por vírgula)
        all_genres_list = []
        df['Genres'].dropna().astype(str).apply(lambda x: all_genres_list.extend(x.split(',')))  # Criamos uma lista com todos os gêneros
        genre_counts = pd.Series(all_genres_list).str.strip().value_counts().head(10)  # Contamos os mais frequentes

        # Criar gráfico de barras horizontal
        fig, ax = plt.subplots(figsize=(8, 5))
        genre_counts.plot(kind='barh', color='skyblue', ax=ax)  # Gráfico horizontal
        ax.set_title("Top 10 Gêneros mais Frequentes")
        ax.set_ylabel("Gêneros")
        ax.set_xlabel("Frequência")
        st.pyplot(fig)  # Exibir o gráfico no Streamlit
    else:
        st.error("A coluna 'Genres' não foi encontrada no dataset.")

# Rodar o Streamlit
if __name__ == "__main__":
    main()
