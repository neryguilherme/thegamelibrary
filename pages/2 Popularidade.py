import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Carregar o arquivo .parquet
path = os.getcwd()
caminho_parquet = os.path.join(path, 'games_cleaned.parquet')

df = pd.read_parquet(caminho_parquet)

# Pré-processamento de dados
df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
df['Price'] = df['Price'].fillna(0)
df['User score'] = df['User score'].fillna(0)
df['Positive'] = df['Positive'].fillna(0)
df['Negative'] = df['Negative'].fillna(0)

# Extrair o primeiro gênero antes da vírgula
df['Primary Genre'] = df['Tags'].str.split(',').str[0]

# Remover categorias indesejadas da coluna 'Primary Genre'
categorias_a_remover = ['Audio Production', 'Accounting', 'Web Publishing', 'Photo Editing', 'Software Training', 'Design & Illustration', 'Utilities', 'Video Production', 'Animation & Modeling', 'Nudity', 'Sexual Content', 'Hacking', 'Games Workshop', 'Free to Play', 'Early access', 'Visual Novel', 'VR', 'Great Soundtrack']
df = df[~df['Primary Genre'].isin(categorias_a_remover)]

# Gráfico 10: Quantidade de Jogos por Gênero com Filtro de Seleção
st.subheader("Quantidade de Jogos por Gênero")

# Filtro de quantidade de gêneros a serem exibidos
num_genres = st.slider("Selecione o número de gêneros a exibir", 1, 100, 10)

# Contar quantos jogos existem por gênero
genre_count = df['Primary Genre'].value_counts().reset_index(name='Game Count')
genre_count.columns = ['Gênero', 'Quantidade de Jogos']

# Ordenar os gêneros de forma decrescente pela quantidade de jogos
genre_count_sorted = genre_count.sort_values(by='Quantidade de Jogos', ascending=False)

# Filtrar para exibir apenas os gêneros com o número de jogos selecionado
genre_count_filtered = genre_count_sorted.head(num_genres)

# Gráfico de barras horizontais
fig_genre_bar = px.bar(genre_count_filtered, 
                       x='Quantidade de Jogos', 
                       y='Gênero', 
                       title="Quantidade de Jogos por Gênero",
                       orientation='h',  # Barra horizontal
                       color='Gênero',
                       color_discrete_sequence=px.colors.qualitative.Set3,
                       labels={'Quantidade de Jogos': 'Quantidade de Jogos', 'Gênero': 'Gênero'})

# Exibir o gráfico
st.plotly_chart(fig_genre_bar)

st.write("""
    Esse gráfico de barras horizontais mostra a quantidade de jogos disponíveis por gênero, proporcionando uma visão geral da popularidade de cada gênero na plataforma. Ele ajuda a direcionar a recomendação para gêneros mais populares, adaptando-se ao gosto do usuário, dependendo da quantidade de jogos em cada categoria.
""")

# Gráfico 5: Comparação de Tempo Médio de Jogo por Gênero
st.subheader("Comparação de Tempo Médio de Jogo por Gênero")

# Filtro de quantidade de gêneros a serem exibidos, com chave única para evitar o erro de ID duplicado
num_genres = st.slider("Selecione o número de gêneros a exibir", 1, 100, 10, key="num_genres_slider")

# Média de tempo de jogo por gênero
time_by_genre = df.groupby('Primary Genre')['Average playtime two weeks'].mean().reset_index()

# Ordenar o DataFrame por tempo médio (decrescente)
time_by_genre = time_by_genre.sort_values(by='Average playtime two weeks', ascending=False)

# Filtrar para exibir apenas os gêneros com o número de gêneros selecionado
time_by_genre_filtered = time_by_genre.head(num_genres)

# Gráfico de barras
fig_time = px.bar(time_by_genre_filtered, 
                  x='Primary Genre', 
                  y='Average playtime two weeks', 
                  title="Tempo Médio de Jogo por Gênero (Ordenado)",
                  labels={'Primary Genre': 'Gênero', 'Average playtime two weeks': 'Tempo Médio (minutos)'},
                  color='Primary Genre', 
                  color_discrete_sequence=px.colors.qualitative.Vivid)

fig_time.update_layout(xaxis_tickangle=-45)

st.plotly_chart(fig_time)

st.write("""
    O gráfico de barras compara o tempo médio de jogo por gênero, o que pode ajudar a recomendar jogos com base no tempo que o usuário deseja investir. Se um usuário prefere jogos rápidos, por exemplo, pode ser orientado para gêneros com um tempo médio de jogo mais curto.
""")