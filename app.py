import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Carregamento de dados
def load_data():
    path = os.getcwd()
    caminho_parquet = os.path.join(path, 'games.parquet')
    
    if not os.path.exists(caminho_parquet):
        st.error("Arquivo games.parquet não encontrado no diretório atual.")
        return pd.DataFrame()

    colunas_necessarias = [
        "Name", "Release date", "Price", "User score", "Genres", 
        "Windows", "Mac", "Linux"
    ]
    
    df = pd.read_parquet(caminho_parquet, columns=colunas_necessarias)
    
    # Pré-processamento
    df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
    df['Price'] = df['Price'].fillna(0)
    df['User score'] = df['User score'].fillna(0)
    df['Genres'] = df['Genres'].fillna("Sem Genero")
    df['Windows'] = df['Windows'].fillna(False).astype(bool)
    df['Mac'] = df['Mac'].fillna(False).astype(bool)
    df['Linux'] = df['Linux'].fillna(False).astype(bool)
    df = df.drop_duplicates(subset=['Name'], keep='first')
    
    return df

def show_data_info(df):
    with st.expander("Informações do Dataset"):
        st.write(df.head())
        st.write("Colunas:", df.columns)
        st.write("Tipos de dados:", df.dtypes)
        st.write("Dados Nulos:", df.isnull().sum())

def filter_data(df):
    st.sidebar.header("Filtros de Dados")

    genero = st.sidebar.multiselect("Selecione os Gêneros", options=df['Genres'].unique())
    preco_max = st.sidebar.slider("Preço Máximo", 0, int(df['Price'].max()), int(df['Price'].max()))
    pontuacao_min = st.sidebar.slider("Pontuação Mínima dos Usuários", 0, int(df['User score'].max()), 10)

    windows = st.sidebar.checkbox("Compatível com Windows", value=True)
    mac = st.sidebar.checkbox("Compatível com Mac", value=False)
    linux = st.sidebar.checkbox("Compatível com Linux", value=False)

    df_filtrado = df.copy()
    if genero:
        df_filtrado = df_filtrado[
            df_filtrado['Genres'].apply(lambda x: any(g in str(x) for g in genero))
        ]

    df_filtrado = df_filtrado[
        (df_filtrado['Price'] <= preco_max) &
        (df_filtrado['User score'] >= pontuacao_min)
    ]

    if windows:
        df_filtrado = df_filtrado[df_filtrado['Windows'] is True]
    if mac:
        df_filtrado = df_filtrado[df_filtrado['Mac'] is True]
    if linux:
        df_filtrado = df_filtrado[df_filtrado['Linux'] is True]
    
    return df_filtrado

def display_price_distribution(df):
    st.write("Distribuição de Preços dos Jogos")
    fig = px.histogram(
        df,
        x='Price',
        nbins=20,
        title="Distribuição de Preços",
        labels={'Price': 'Preço'},
        marginal="box",
        color_discrete_sequence=['skyblue']
    )
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig)

def display_user_scores_distribution(df):
    st.write("Distribuição das Pontuações dos Usuários")
    fig = px.histogram(
        df,
        x='User score',
        nbins=20,
        title="Distribuição das Pontuações",
        labels={'User score': 'Pontuação'},
        marginal="box",
        color_discrete_sequence=['purple']
    )
    st.plotly_chart(fig)

def display_price_vs_user_score(df):
    st.write("Relação entre Preço e Pontuação dos Usuários")
    fig = px.scatter(
        df,
        x='Price',
        y='User score',
        color='Genres',
        size='Price',
        title="Preço x Pontuação",
        labels={'Price': 'Preço', 'User score': 'Pontuação'}
    )
    st.plotly_chart(fig)

def display_games_launch_over_time(df):
    st.write("Lançamentos de Jogos ao Longo do Tempo")
    df = df[df['Release date'].notnull()]
    df = df[df['Release date'].dt.year <= 2024]
    
    df_lancamento = df.groupby(df['Release date'].dt.year).size().reset_index(name='Quantidade')
    
    fig = px.bar(
        df_lancamento,
        x='Release date',
        y='Quantidade',
        title="Quantidade de Jogos por Ano até 2024",
        labels={'Release date': 'Ano', 'Quantidade': 'Quantidade de Jogos'},
        color='Quantidade',
        color_continuous_scale='Blues'
    )
    
    st.plotly_chart(fig)

def display_top_genres(df):
    st.write("Top 10 Gêneros Mais Populares")
    generos_populares = df['Genres'].value_counts().head(10).reset_index()
    generos_populares.columns = ['Gênero', 'Quantidade']
    fig = px.bar(
        generos_populares,
        x='Quantidade',
        y='Gênero',
        orientation='h',
        title="Top 10 Gêneros",
        color='Quantidade',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig)

def main():    
    st.title("Análise de Dados de Jogos da Steam")
    st.write("Aplicativo para explorar e analisar os dados de jogos da Steam.")
    
    df = load_data()
    if df.empty:
        st.stop()
    
    show_data_info(df)
    df_filtrado = filter_data(df)

    st.write("Total de Jogos Encontrados:", len(df_filtrado))
    st.dataframe(df_filtrado[['Name', 'Release date', 'Price', 'User score', 'Genres']].head(50))
    
    st.subheader("Análises Gráficas")
    display_price_distribution(df_filtrado)
    display_user_scores_distribution(df_filtrado)
    display_price_vs_user_score(df_filtrado)
    display_games_launch_over_time(df_filtrado)
    display_top_genres(df_filtrado)

if __name__ == "__main__":
    main()
