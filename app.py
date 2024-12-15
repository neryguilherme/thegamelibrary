import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Configurações do Streamlit
st.title("Análise de Dados de Jogos da Steam")
st.write("Aplicativo para explorar e analisar os dados de jogos da Steam.")

# Carregar o arquivo .parquet
path = os.getcwd()
caminho_parquet = os.path.join(path, 'games.parquet')

df = pd.read_parquet(caminho_parquet)

# Pré-processamento de dados
df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
df['Price'] = df['Price'].fillna(0)
df['User score'] = df['User score'].fillna(0)
df['Positive'] = df['Positive'].fillna(0)
df['Negative'] = df['Negative'].fillna(0)

# Extrair o primeiro gênero antes da vírgula
df['Primary Genre'] = df['Genres'].str.split(',').str[0]

# Exibir informações do dataset
if st.checkbox("Mostrar informações do dataset"):
    st.write(df.head())
    st.write("Colunas:", df.columns)
    st.write("Tipos de dados:", df.dtypes)
    st.write("Dados Nulos:", df.isnull().sum())

# Gráfico 1: Distribuição de Preços dos Jogos por Gênero
st.subheader("Distribuição de Preços dos Jogos por Preço")

# Criar faixas de preço
bins = [0, 10, 30, 60, 100, 200, float('inf')]
labels = ['$0-$10', '$10-$30', '$30-$60', '$60-$100', '$100-$200', '$200+']
df['Price Range'] = pd.cut(df['Price'], bins=bins, labels=labels, right=False)

# Filtro de preço
price_filter = st.slider("Selecione a faixa de preço", 0, 200, (0, 100), step=10)
df_filtered_price = df[(df['Price'] >= price_filter[0]) & (df['Price'] <= price_filter[1])]

# Contagem de jogos por gênero e faixa de preço
price_dist = df_filtered_price.groupby(['Primary Genre', 'Price Range']).size().reset_index(name='Game Count')

# Gráfico de pizza interativo
fig = px.pie(price_dist, names='Price Range', values='Game Count', 
             color='Price Range', 
             title="Distribuição de Preços dos Jogos ",
             color_discrete_map={ 
                 '$0-$10': 'skyblue', '$10-$30': 'lightgreen', '$30-$60': 'orange', 
                 '$60-$100': 'pink', '$100-$200': 'yellow', '$200+': 'red'
             },
             template="plotly_dark", 
             hole=0.3)

st.plotly_chart(fig)

# Gráfico 2: Distribuição de Notas dos Jogos por Gênero

st.subheader("Distribuição de Notas dos Jogos por Gênero")

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

# Separar o gênero principal
df['Primary Genre'] = df['Genres'].str.split(',').str[0]

# Média de estrelas por gênero
genre_stars = df.groupby('Primary Genre')['Stars'].mean().reset_index()

# Filtro de intervalo de notas
star_filter = st.slider("Selecione a faixa de estrelas", 0, 5, (0, 5))

# Filtrar por intervalo de estrelas
filtered_genre_stars = genre_stars[(genre_stars['Stars'] >= star_filter[0]) & (genre_stars['Stars'] <= star_filter[1])]

# Gráfico de barras agrupadas para médias de notas por gênero
fig2 = px.bar(filtered_genre_stars, x='Primary Genre', y='Stars',
              color='Primary Genre', 
              title="Média de Notas por Gênero",
              labels={'Primary Genre': 'Gênero', 'Stars': 'Média de Estrelas'},
              color_discrete_sequence=px.colors.qualitative.Set2)

fig2.update_layout(barmode='group', xaxis_tickangle=-45)

st.plotly_chart(fig2)

# NOVOS GRÁFICOS

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

# Gráfico 4: Distribuição de Jogos por Faixa Etária Requerida
st.subheader("Distribuição de Jogos por Faixa Etária Requerida")

# Filtro de faixa etária
age_filter = st.slider("Selecione a faixa etária requerida", int(df['Required age'].min()), int(df['Required age'].max()), (int(df['Required age'].min()), int(df['Required age'].max())))

# Filtrando por faixa etária
df_filtered_age = df[(df['Required age'] >= age_filter[0]) & (df['Required age'] <= age_filter[1])]

# Contagem de jogos por faixa etária
age_dist = df_filtered_age['Required age'].value_counts().reset_index(name='Game Count')
age_dist.columns = ['Faixa Etária Requerida', 'Quantidade de Jogos']

# Gráfico de barras
fig_age = px.bar(age_dist, x='Faixa Etária Requerida', y='Quantidade de Jogos', 
                 title="Distribuição de Jogos por Faixa Etária Requerida",
                 labels={'Faixa Etária Requerida': 'Faixa Etária', 'Quantidade de Jogos': 'Quantidade de Jogos'},
                 color='Faixa Etária Requerida')

st.plotly_chart(fig_age)

# Gráfico 5: Comparação de Tempo Médio de Jogo por Gênero
st.subheader("Comparação de Tempo Médio de Jogo por Gênero")

# Filtro de tempo de jogo
time_filter = st.slider("Selecione o intervalo de tempo de jogo (em minutos)", 0, int(df['Average playtime forever'].max()), (0, int(df['Average playtime forever'].max())))

# Filtrando por tempo de jogo
df_filtered_time = df[(df['Average playtime forever'] >= time_filter[0]) & (df['Average playtime forever'] <= time_filter[1])]

# Média de tempo de jogo por gênero
time_by_genre = df_filtered_time.groupby('Primary Genre')['Average playtime forever'].mean().reset_index()

# Gráfico de barras
fig_time = px.bar(time_by_genre, x='Primary Genre', y='Average playtime forever', 
                  title="Tempo Médio de Jogo por Gênero",
                  labels={'Primary Genre': 'Gênero', 'Average playtime forever': 'Tempo Médio de Jogo (minutos)'})

fig_time.update_layout(barmode='group', xaxis_tickangle=-45)

st.plotly_chart(fig_time)
