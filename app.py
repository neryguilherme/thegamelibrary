import streamlit as st
import pandas as pd
import plotly.express as px
import os




# Configurações do Streamlit
st.title("Análise de Dados de Jogos da Steam")
st.write("Aplicativo para explorar e analisar os dados de jogos da Steam.")

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
categorias_a_remover = ['Audio Production', 'Accounting', 'Web Publishing', 'Photo Editing', 'Software Training', 'Design & Illustration', 'Utilities', 'Video Production', 'Animation & Modeling', 'Nudity', 'Sexual Content', 'Hacking', 'Games Workshop', 'Free to Play', 'Early access', 'Visual Novel', 'VR']
df = df[~df['Primary Genre'].isin(categorias_a_remover)]

# Exibir informações do dataset
if st.checkbox("Mostrar informações do dataset"):
    st.write(df.head())
    st.write("Colunas:", df.columns)
    st.write("Tipos de dados:", df.dtypes)
    st.write("Dados Nulos:", df.isnull().sum())
    
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

# Gráfico 2: Distribuição de Notas dos Jogos por Gênero

# st.subheader("Distribuição de Notas dos Jogos por Gênero")

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

# df['Genre'] = df['Genres'].str.split(',').str[0]


# Calcular as estrelas para cada jogo
df['Stars'] = df.apply(calculate_stars, axis=1)

# Média de estrelas por gênero
# genre_stars = df.groupby('Primary genre')['Stars'].mean().reset_index()

# # Ordenar por média de estrelas (de forma decrescente)
# genre_stars = genre_stars.sort_values(by='Stars', ascending=False)

# # Filtro de intervalo de notas
# star_filter = st.slider("Selecione a faixa de estrelas", 0, 5, (0, 5))

# # Filtrar por intervalo de estrelas
# filtered_genre_stars = genre_stars[(genre_stars['Stars'] >= star_filter[0]) & (genre_stars['Stars'] <= star_filter[1])]

# # Gráfico de barras agrupadas para médias de notas por gênero
# fig2 = px.bar(filtered_genre_stars, x='Genre', y='Stars',
#               color='Genre', 
#               title="Média de Notas por Gênero",
#               labels={'Genre': 'Gênero', 'Stars': 'Média de Estrelas'},
#               color_discrete_sequence=px.colors.qualitative.Set2)

# fig2.update_layout(barmode='group', xaxis_tickangle=-45)

# st.plotly_chart(fig2)

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

st.write("""
    Este gráfico de barras exibe a quantidade de jogos disponíveis por sistema operacional (Windows, Mac e Linux). Ele é útil para usuários que têm preferências por sistemas específicos, podendo recomendar jogos compatíveis com o sistema operacional de sua escolha.
""")

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


# # Gráfico 6: Tendência de Lançamentos de Categorias ao Longo do Tempo
# st.subheader("Tendência de Lançamentos de Categorias ao Longo do Tempo")

# # Criar uma coluna com o ano de lançamento
# df['Release Year'] = df['Release date'].dt.year

# df = df[df['Release Year'] >= 2010]

# # Intervalo de anos disponível no dataset
# min_year = int(df['Release Year'].min())
# max_year = int(df['Release Year'].max())

# # Filtro de intervalo de anos
# year_filter = st.slider("Selecione o intervalo de anos de lançamento", 
#                         min_year, max_year, (min_year, max_year))

# # Filtrar o DataFrame pelo intervalo de anos selecionado
# df_filtered_year = df[(df['Release Year'] >= year_filter[0]) & (df['Release Year'] <= year_filter[1])]

# # Contar o número de lançamentos por ano e categoria
# category_trend = df_filtered_year.groupby(['Release Year', 'Primary Genre']).size().reset_index(name='Game Count')

# # Gráfico de área interativo
# fig_trend_area = px.area(category_trend, x='Release Year', y='Game Count', 
#                          color='Primary Genre',
#                          title="Tendência de Lançamentos de Categorias ao Longo do Tempo",
#                          labels={'Release Year': 'Ano de Lançamento', 'Game Count': 'Quantidade de Jogos'},
#                          markers=True)

# fig_trend_area.update_layout(xaxis=dict(dtick=1))  # Ajustar para exibir todos os anos no eixo X

# # Exibir o gráfico
# st.plotly_chart(fig_trend_area)

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


# Gráfico 8: Contagem de Jogos por Idioma Suportado (Português/Português Brasileiro)
df.columns = df.columns.str.lower()



# Gráfico 10: Quantidade de Jogos por Gênero com Filtro de Seleção
st.subheader("Quantidade de Jogos por Gênero")

# Filtro de quantidade de gêneros a serem exibidos
num_genres = st.slider("Selecione o número de gêneros a exibir", 1, 100, 10)

# Contar quantos jogos existem por gênero
genre_count = df['primary genre'].value_counts().reset_index(name='Game Count')
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

# Gráfico 9: Gêneros mais Populares por Plataforma (Windows, Mac, Linux)
st.subheader("Gêneros Mais Populares por Plataforma")

# Filtro de plataforma
platform_filter = st.selectbox("Selecione a plataforma", ['Windows', 'Mac', 'Linux'], index=0)

# Filtrando os dados conforme a plataforma selecionada
platform_column = platform_filter.lower()  # Nome da coluna na tabela (Windows, Mac, Linux)
df_filtered_platform = df[df[platform_column] == 1]

# Contagem de jogos por gênero
genre_platform_count = df_filtered_platform.groupby('primary genre').size().reset_index(name='Game Count')

# Filtrar apenas gêneros com mais de 95 jogos
genre_platform_count_filtered = genre_platform_count[genre_platform_count['Game Count'] > 95]

# Ordenar os gêneros por quantidade de jogos
genre_platform_count_sorted = genre_platform_count_filtered.sort_values(by='Game Count', ascending=False)

# Gráfico de barras empilhadas para gêneros por plataforma
fig_platform = px.bar(genre_platform_count_sorted, 
                      x='primary genre', 
                      y='Game Count', 
                      title=f"Gêneros mais Populares no {platform_filter}",
                      labels={'primary genre': 'Gênero', 'Game Count': 'Quantidade de Jogos'},
                      color='primary genre', 
                      color_discrete_sequence=px.colors.qualitative.Set2)

fig_platform.update_layout(xaxis_tickangle=-45)

st.plotly_chart(fig_platform)

st.write("""
    Este gráfico de barras interativo exibe quais gêneros são mais populares em diferentes plataformas (Windows, Mac, Linux). Ele pode ser usado para recomendar jogos baseados não apenas no gênero preferido, mas também na plataforma escolhida pelo usuário.
""")

# # Novo Gráfico: Jogos mais Populares por Gênero com Filtro
# st.subheader("Jogos mais Populares por Gênero")

# # Filtro de número de jogos a serem exibidos
# num_jogos_populares = st.slider("Selecione o número de jogos mais populares a exibir", 1, 50, 10)

# # Contagem de jogos por gênero
# genre_count = df['primary genre'].value_counts().reset_index(name='Game Count')
# genre_count.columns = ['Gênero', 'Quantidade de Jogos']

# # Ordenar os gêneros de forma decrescente pela quantidade de jogos
# genre_count_sorted = genre_count.sort_values(by='Quantidade de Jogos', ascending=False)

# # Filtrar para exibir apenas os gêneros com o número de jogos selecionado
# genre_count_filtered = genre_count_sorted.head(num_jogos_populares)

# # Gráfico de dispersão para quantidade de jogos por gênero
# fig_genre_popular = px.scatter(genre_count_filtered, 
#                                x='Gênero', 
#                                y='Quantidade de Jogos', 
#                                title="Jogos Mais Populares por Gênero",
#                                labels={'Quantidade de Jogos': 'Quantidade de Jogos', 'Gênero': 'Gênero'},
#                                color='Gênero', 
#                                color_discrete_sequence=px.colors.qualitative.Set3)

# fig_genre_popular.update_layout(
#     xaxis_tickangle=-45, 
#     template="plotly_dark", 
#     showlegend=False
# )

# st.plotly_chart(fig_genre_popular)

# Título da seção
st.subheader("Distribuição de Jogos por Idioma Suportado")

# Pré-processamento: separar os idiomas em uma lista
df['Processed Languages'] = df['supported languages'].str.split(',').apply(lambda x: [lang.strip().lower() for lang in x] if isinstance(x, list) else [])

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





