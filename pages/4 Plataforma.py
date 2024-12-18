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

# Gráfico 3: Distribuição de Jogos por Sistema Operacional Suportado
st.subheader("Distribuição de Jogos por Sistema Operacional Suportado")

# Filtro de sistema operacional
os_filter = st.multiselect("Selecione os sistemas operacionais", ['Windows', 'Mac', 'Linux'], default=['Windows', 'Mac', 'Linux'])

# Filtrando os dados conforme os sistemas operacionais selecionados
df_filtered_os = df[df[os_filter].any(axis=1)]

# Contagem de jogos por sistema operacional
os_dist = df_filtered_os[os_filter].sum().reset_index(name='Game Count')
os_dist.columns = ['Sistema Operacional', 'Quantidade de Jogos']

# Gráfico de barras interativo
fig_os = px.bar(os_dist, x='Sistema Operacional', y='Quantidade de Jogos', 
                title="Distribuição de Jogos por Sistema Operacional Suportado", 
                color='Sistema Operacional', 
                color_discrete_map={'Windows': 'blue', 'Mac': 'green', 'Linux': 'red'},
                labels={'Sistema Operacional': 'Sistema Operacional', 'Quantidade de Jogos': 'Quantidade de Jogos'})

st.plotly_chart(fig_os)

st.write("""
    Este gráfico de barras exibe a quantidade de jogos disponíveis por sistema operacional (Windows, Mac e Linux). Ele é útil para usuários que têm preferências por sistemas específicos, podendo recomendar jogos compatíveis com o sistema operacional de sua escolha.
""")



# Gráfico 9: Gêneros mais Populares por Plataforma (Windows, Mac, Linux)
st.subheader("Gêneros Mais Populares por Plataforma")

# Filtro de plataforma
platform_filter = st.selectbox("Selecione a plataforma", ['Windows', 'Mac', 'Linux'], index=0)

# Filtrando os dados conforme a plataforma selecionada
platform_column = platform_filter  # Nome da coluna na tabela (Windows, Mac, Linux)
df_filtered_platform = df[df[platform_column] == 1]

# Contagem de jogos por gênero
genre_platform_count = df_filtered_platform.groupby('Primary Genre').size().reset_index(name='Game Count')

# Filtrar apenas gêneros com mais de 95 jogos
genre_platform_count_filtered = genre_platform_count[genre_platform_count['Game Count'] > 95]

# Ordenar os gêneros por quantidade de jogos
genre_platform_count_sorted = genre_platform_count_filtered.sort_values(by='Game Count', ascending=False)

# Gráfico de barras empilhadas para gêneros por plataforma
fig_platform = px.bar(genre_platform_count_sorted, 
                      x='Primary Genre', 
                      y='Game Count', 
                      title=f"Gêneros mais Populares no {platform_filter}",
                      labels={'Primary Genre': 'Gênero', 'Game Count': 'Quantidade de Jogos'},
                      color='Primary Genre', 
                      color_discrete_sequence=px.colors.qualitative.Set2)

fig_platform.update_layout(xaxis_tickangle=-45)

st.plotly_chart(fig_platform)

st.write("""
    Este gráfico de barras interativo exibe quais gêneros são mais populares em diferentes plataformas (Windows, Mac, Linux). Ele pode ser usado para recomendar jogos baseados não apenas no gênero preferido, mas também na plataforma escolhida pelo usuário.
""")