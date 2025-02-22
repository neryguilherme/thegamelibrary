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

# Função para calcular a nota em estrelas
def calculate_stars(row):
    total_reviews = row['Positive'] + row['Negative']
    if total_reviews == 0:
        return 0  # Se não houver avaliações, atribui 0 estrelas
    positive_percentage = (row['Positive'] / total_reviews) * 100
    
    if positive_percentage >= 80:
        return 5
    elif positive_percentage >= 60:
        return 4
    elif positive_percentage >= 40:
        return 3
    elif positive_percentage >= 20:
        return 2
    elif positive_percentage >= 1:
        return 1
    else:
        return 0


# Calcular as estrelas para cada jogo
df['Stars'] = df.apply(calculate_stars, axis=1)


# Categorizar jogos com base na faixa etária requerida
df['Age Category'] = df['Required age'].apply(lambda x: 'Com restrição de idade' if x > 0 else 'Sem restrição de idade')

# Contagem de jogos por categoria de faixa etária
age_count = df['Age Category'].value_counts().reset_index(name='Game Count')
age_count.columns = ['Faixa Etária', 'Quantidade de Jogos']

# Gráfico de contagem
fig_age_count = px.bar(age_count, 
                       x='Faixa Etária', 
                       y='Quantidade de Jogos', 
                       title="Contagem de Jogos por Faixa Etária Requerida",
                       labels={'Faixa Etária': 'Faixa Etária Requerida', 'Quantidade de Jogos': 'Quantidade de Jogos'},
                       color='Faixa Etária', 
                       color_discrete_map={'Com restrição': 'lightblue', 'Sem restrição ': 'lightgreen'})

st.plotly_chart(fig_age_count)



# Gráfico 4: Distribuição de Jogos por Faixa Etária Requerida
st.subheader("Distribuição de Jogos por Faixa Etária Requerida")

# Filtro de faixa etária
age_filter = st.slider("Selecione a faixa etária requerida", int(df['Required age'].min()), int(df['Required age'].max()), (int(df['Required age'].min()), int(df['Required age'].max())))

# Filtrando por faixa etária maior que 0
df_filtered_age = df[(df['Required age'] > 0) & 
                     (df['Required age'] >= age_filter[0]) & 
                     (df['Required age'] <= age_filter[1])]

# Contagem de jogos por faixa etária
age_dist = df_filtered_age['Required age'].value_counts().reset_index(name='Game Count')
age_dist.columns = ['Faixa Etária Requerida', 'Quantidade de Jogos']

# Gráfico de pizza
fig_age_pie = px.pie(age_dist, names='Faixa Etária Requerida', values='Quantidade de Jogos',
                     title="Distribuição de Jogos por Faixa Etária Requerida",
                     labels={'Faixa Etária Requerida': 'Faixa Etária', 'Quantidade de Jogos': 'Quantidade de Jogos'},
                     color='Faixa Etária Requerida')

st.plotly_chart(fig_age_pie)

st.write("""
    Com este gráfico de pizza, é possível visualizar a quantidade de jogos disponíveis para diferentes faixas etárias. Ele é útil para restringir a recomendação de jogos com base na faixa etária adequada, garantindo que os usuários não recebam sugestões de jogos inadequados para sua idade.a.
""")