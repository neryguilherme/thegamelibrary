import kagglehub
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# Configurações do Streamlit
st.title("Análise de Dados de Jogos da Steam")
st.write("Aplicativo para explorar e analisar os dados de jogos da Steam.")


path = kagglehub.dataset_download("mexwell/steamgames")
caminho_csv = os.path.join(path, 'games.csv')


df = pd.read_csv(caminho_csv)

# Pré-processamento de dados
df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
df['Price'] = df['Price'].fillna(0)
df['User score'] = df['User score'].fillna(0)

# Exibir informações do dataset
if st.checkbox("Mostrar informações do dataset"):
    st.write(df.head())
    st.write("Colunas:", df.columns)
    st.write("Tipos de dados:", df.dtypes)
    st.write("Dados Nulos:", df.isnull().sum())

# Filtros de usuário
st.sidebar.header("Opções de Filtragem")
genero = st.sidebar.multiselect("Selecione os Gêneros", options=df['Genres'].dropna().unique())
preco_max = st.sidebar.slider("Preço Máximo", 0, int(df['Price'].max()), 50)
pontuacao_min = st.sidebar.slider("Pontuação Mínima dos Usuários", 0, int(df['User score'].max()), 70)
windows = st.sidebar.checkbox("Compatível com Windows", value=True)
mac = st.sidebar.checkbox("Compatível com Mac", value=False)
linux = st.sidebar.checkbox("Compatível com Linux", value=False)

# Converte valores NaN na coluna 'Genres' para strings vazias
df['Genres'] = df['Genres'].fillna("")

# Aplicar filtros
df_filtrado = df.copy()
if genero:
    df_filtrado = df_filtrado[df_filtrado['Genres'].apply(lambda x: any(g in x for g in genero))]
df_filtrado = df_filtrado[(df_filtrado['Price'] <= preco_max) & 
                          (df_filtrado['User score'] >= pontuacao_min) & 
                          (df_filtrado['Windows'] == windows) & 
                          (df_filtrado['Mac'] == mac) & 
                          (df_filtrado['Linux'] == linux)]

# Exibir dados filtrados
st.write("Total de Jogos Encontrados:", len(df_filtrado))
st.dataframe(df_filtrado[['Name', 'Release date', 'Price', 'User score', 'Genres']])

# Gráficos de análise
st.subheader("Análises Gráficas")

# Gráfico 1: Distribuição de Preços
st.write("Distribuição de Preços dos Jogos")
fig1 = px.histogram(df_filtrado, x='Price', nbins=20, title="Distribuição de Preços dos Jogos", 
                    labels={'Price': 'Preço'}, marginal="box", color_discrete_sequence=['skyblue'])
fig1.update_layout(bargap=0.1)
st.plotly_chart(fig1)

# Gráfico 2: Distribuição de Pontuação dos Usuários
st.write("Distribuição das Pontuações dos Usuários")
fig2 = px.histogram(df_filtrado, x='User score', nbins=20, title="Distribuição das Pontuações dos Usuários", 
                    labels={'User score': 'Pontuação dos Usuários'}, marginal="box", color_discrete_sequence=['purple'])
st.plotly_chart(fig2)

# Gráfico 3: Relação entre Preço e Pontuação dos Usuários
st.write("Relação entre Preço e Pontuação dos Usuários")
fig3 = px.scatter(df_filtrado, x='Price', y='User score', color='Genres', size='Price', 
                  title="Relação entre Preço e Pontuação dos Usuários", 
                  labels={'Price': 'Preço', 'User score': 'Pontuação dos Usuários'})
st.plotly_chart(fig3)

# Gráfico 4: Lançamento de Jogos ao Longo do Tempo
st.write("Lançamento de Jogos ao Longo do Tempo")
df_lancamento = df_filtrado.groupby(df_filtrado['Release date'].dt.year).size().reset_index(name='Quantidade')
fig4 = px.area(df_lancamento, x='Release date', y='Quantidade', title="Lançamento de Jogos ao Longo do Tempo", 
               labels={'Release date': 'Ano', 'Quantidade': 'Quantidade de Jogos'}, color_discrete_sequence=['orange'])
st.plotly_chart(fig4)

# Gráfico 5: Gêneros Mais Populares
st.write("Gêneros Mais Populares")
generos_populares = df_filtrado['Genres'].value_counts().head(10).reset_index()
generos_populares.columns = ['Gênero', 'Quantidade']
fig5 = px.bar(generos_populares, x='Quantidade', y='Gênero', orientation='h', 
              title="Top 10 Gêneros Mais Populares", 
              labels={'Quantidade': 'Quantidade de Jogos', 'Gênero': 'Gêneros'}, color='Quantidade', 
              color_continuous_scale='viridis')
st.plotly_chart(fig5)
