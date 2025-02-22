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



# Gráfico 1: Distribuição de Preços dos Jogos por Gênero
st.subheader("Distribuição de Preços dos Jogos")

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

st.write("""
    Este gráfico de pizza mostra a distribuição de jogos da Steam dentro de diferentes faixas de preço, permitindo visualizar como os jogos se distribuem entre valores baixos e altos. Com ele, podemos compreender a variação no preço dos jogos, o que pode ajudar a personalizar recomendações para usuários com orçamentos específicos, direcionando-os para jogos dentro da sua faixa de preço preferida.
""")


# Gráfico 7: Correlação entre Preço Médio e Avaliações por Gênero
st.subheader("Correlação entre Preço Médio e Avaliações por Gênero")

# Filtro de quantidade de gêneros a exibir
max_genres = len(df['Primary Genre'].unique())  # Número máximo de gêneros
num_genres_filter = st.slider(
    "Selecione o número de gêneros a exibir",
    min_value=1, max_value=150, value=30
)

# Calcular o preço médio, a média de estrelas e a contagem de jogos por gênero
genre_summary = df.groupby('Primary Genre').agg(
    Avg_Price=('Price', 'mean'),  # Preço médio por gênero
    Avg_Stars=('Stars', 'mean'),  # Média de estrelas por gênero
    Game_Count=('Primary Genre', 'size')  # Contagem de jogos por gênero
).reset_index()

# Filtrar apenas gêneros com mais de 200 jogos
genre_summary_filtered = genre_summary[genre_summary['Game_Count'] > 200]

# Ordenar os gêneros pela média de estrelas de forma decrescente
genre_summary_sorted = genre_summary_filtered.sort_values(by='Avg_Stars', ascending=False)

# Selecionar os gêneros com as maiores médias de estrelas, conforme a quantidade escolhida pelo usuário
genre_summary_top = genre_summary_sorted.head(num_genres_filter)

# Criar gráfico de dispersão para a correlação entre preço e estrelas
fig_scatter = px.scatter(
    genre_summary_top,
    x='Avg_Price',  # Preço médio
    y='Avg_Stars',  # Avaliações médias
    size='Game_Count',  # Tamanho dos pontos com base na quantidade de jogos
    color='Primary Genre',  # Cor dos pontos de acordo com o gênero
    title="Correlação entre Preço Médio e Avaliações por Gênero",
    labels={'Avg_Price': 'Preço Médio (USD)', 'Avg_Stars': 'Avaliações (Estrelas - Média)', 'Game_Count': 'Quantidade de Jogos'},
    hover_name='Primary Genre',  # Exibir o nome do gênero ao passar o mouse
    size_max=30  # Limitar o tamanho máximo dos pontos
)

# Ajustar layout do gráfico
fig_scatter.update_layout(
    xaxis_title="Preço Médio (USD)",
    yaxis_title="Avaliações (Estrelas)",
    xaxis=dict(dtick=5)  # Ajustar as marcas no eixo X
)

# Exibir o gráfico
st.plotly_chart(fig_scatter)

st.write("""
    Este gráfico de dispersão exibe a relação entre o preço médio dos jogos e suas avaliações médias por gênero. Ele é valioso para entender como o preço pode influenciar a qualidade percebida, ajudando a recomendar jogos que oferecem um bom equilíbrio entre preço e avaliações positivas, atendendo ao orçamento e às expectativas do usuário.
""")