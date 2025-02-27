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

# Título da seção
st.subheader("Distribuição de Jogos por Idioma Suportado")

# Pré-processamento: separar os idiomas em uma lista
df['Processed Languages'] = df['Supported languages'].str.split(',').apply(lambda x: [lang.strip().lower() for lang in x] if isinstance(x, list) else [])

# Normaliza os idiomas para formas padronizadas
df['Processed Languages'] = df['Processed Languages'].apply(lambda x: ['english' if lang in ['english', "['english']", "['english'"] else
                                                                          'spanish' if lang in ['spanish', 'spanish (mexican)', 'spanish (latin america)', 'spanish (castilian)'] else
                                                                          'portuguese' if lang in ['portuguese', 'Brazil'] else lang for lang in x])

# Explodimos a coluna para criar uma linha para cada idioma
df_languages_exploded = df.explode('Processed Languages')

# Contamos a frequência de cada idioma
language_counts = df_languages_exploded['Processed Languages'].value_counts().reset_index()
language_counts.columns = ['Idioma', 'Quantidade de Jogos']

# Seleção da quantidade de idiomas que o usuário deseja visualizar
num_languages = st.slider("Selecione a quantidade de idiomas a serem exibidos", min_value=1, max_value=10, value=5)

# Filtra os idiomas mais comuns com base na quantidade selecionada
top_languages = language_counts.head(num_languages)

# Exibe os idiomas mais comuns
st.write("Idiomas encontrados:", top_languages)

# Gráfico interativo (exibindo os idiomas mais comuns)
fig_languages = px.pie(top_languages, 
                       names='Idioma', 
                       values='Quantidade de Jogos', 
                       title="Distribuição de Jogos por Idioma Suportado",
                       hole=0.4)

st.plotly_chart(fig_languages)

st.write("""
    O gráfico de barras mostra a quantidade de jogos suportados em diferentes idiomas. Esse gráfico é útil para recomendar jogos que possuem a linguagem preferida do usuário, como português, por exemplo, oferecendo uma experiência mais personalizada e acessível.
""")
