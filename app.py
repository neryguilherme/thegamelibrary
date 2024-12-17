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
df['Primary Genre'] = df['Genres'].str.split(',').str[0]

# Remover categorias indesejadas da coluna 'Primary Genre'
categorias_a_remover = ['Audio Production', 'Accounting', 'Web Publishing', 'Photo Editing', 'Software Training', 'Design & Illustration', 'Utilities', 'Video Production', 'Animation & Modeling']
df = df[~df['Primary Genre'].isin(categorias_a_remover)]

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
                                    labels={'Primary Genre': 'Gênero', 'Average playtime forever': 'Tempo Médio (minutos)'},
                  color='Primary Genre', 
                  color_discrete_sequence=px.colors.qualitative.Vivid)

# Ordenar o DataFrame por tempo médio (decrescente)
time_by_genre = time_by_genre.sort_values(by='Average playtime forever', ascending=False)

fig_time = px.bar(time_by_genre, 
                  x='Primary Genre', 
                  y='Average playtime forever', 
                  title="Tempo Médio de Jogo por Gênero (Ordenado)",
                  labels={'Primary Genre': 'Gênero', 'Average playtime forever': 'Tempo Médio (minutos)'},
                  color='Primary Genre', 
                  color_discrete_sequence=px.colors.qualitative.Vivid)

fig_time.update_layout(xaxis_tickangle=-45)

st.plotly_chart(fig_time)

# Gráfico 6: Tendência de Lançamentos de Categorias ao Longo do Tempo
st.subheader("Tendência de Lançamentos de Categorias ao Longo do Tempo")

# Criar uma coluna com o ano de lançamento
df['Release Year'] = df['Release date'].dt.year

# Intervalo de anos disponível no dataset
min_year = int(df['Release Year'].min())
max_year = int(df['Release Year'].max())

# Filtro de intervalo de anos
year_filter = st.slider("Selecione o intervalo de anos de lançamento", 
                        min_year, max_year, (min_year, max_year))

# Filtrar o DataFrame pelo intervalo de anos selecionado
df_filtered_year = df[(df['Release Year'] >= year_filter[0]) & (df['Release Year'] <= year_filter[1])]

# Contar o número de lançamentos por ano e categoria
category_trend = df_filtered_year.groupby(['Release Year', 'Primary Genre']).size().reset_index(name='Game Count')

# Gráfico de linha interativo
fig_trend = px.line(category_trend, x='Release Year', y='Game Count', 
                    color='Primary Genre',
                    title="Tendência de Lançamentos de Categorias ao Longo do Tempo",
                    labels={'Release Year': 'Ano de Lançamento', 'Game Count': 'Quantidade de Jogos'},
                    markers=True)

fig_trend.update_layout(xaxis=dict(dtick=1))  # Ajustar para exibir todos os anos no eixo X

st.plotly_chart(fig_trend)

# Gráfico 7: Preço Médio x Avaliações (Média de Estrelas) por Gênero
st.subheader("Correlação entre Preço Médio e Avaliações por Gênero")

# Filtro de intervalo de preços
min_price = float(df['Price'].min())
max_price = float(df['Price'].max())
price_filter = st.slider(
    "Selecione o intervalo de preços",
    min_price, max_price, (min_price, max_price)
)

# Filtrar o DataFrame pelo intervalo de preços selecionado
df_filtered_price = df[(df['Price'] >= price_filter[0]) & (df['Price'] <= price_filter[1])]

# Calcular o preço médio e a média de estrelas por gênero no DataFrame filtrado
genre_summary = df_filtered_price.groupby('Primary Genre').agg(
    Avg_Price=('Price', 'mean'),  # Preço médio por gênero
    Avg_Stars=('Stars', 'mean')  # Média de estrelas por gênero
).reset_index()

# Gráfico de dispersão (Preço Médio x Média de Estrelas por Gênero)
fig_scatter = px.scatter(
    genre_summary,
    x='Avg_Price',
    y='Avg_Stars',
    color='Primary Genre',  # Cor por gênero
    size='Avg_Stars',  # Tamanho dos pontos baseado na média de estrelas
    hover_name='Primary Genre',  # Nome do gênero no hover
    title="Correlação entre Preço Médio e Avaliações por Gênero",
    labels={'Avg_Price': 'Preço Médio (USD)', 'Avg_Stars': 'Avaliações (Estrelas - Média)'}
)

# Ajustar layout do gráfico
fig_scatter.update_layout(
    xaxis_title="Preço Médio (USD)",
    yaxis_title="Avaliações (Estrelas - Média)",
    xaxis=dict(dtick=5)  # Ajuste das marcas no eixo X
)

# Exibir o gráfico
st.plotly_chart(fig_scatter)

# Unir os idiomas português e brasileiro
df['Full audio languages'] = df['Full audio languages'].fillna('')
df['Has Portuguese Audio'] = df['Full audio languages'].apply(
    lambda x: 'Portuguese' in x or 'Brazilian Portuguese' in x
)

# Contar o número de jogos com áudio completo em português
audio_count = df['Has Portuguese Audio'].value_counts().reset_index()
audio_count.columns = ['Audio Language', 'Game Count']

# Gráfico de barras interativo
fig = px.bar(audio_count, x='Audio Language', y='Game Count',
             title="Distribuição de Jogos com Áudio Completo em Português ou Brasileiro",
             labels={'Audio Language': 'Idioma de Áudio', 'Game Count': 'Quantidade de Jogos'},
             color='Audio Language')

st.plotly_chart(fig)

# Gráfico 8: Contagem de Jogos por Idioma Suportado (Português/Português Brasileiro)
df.columns = df.columns.str.lower()

st.subheader("Contagem de Jogos por Idioma Suportado (Português ou Brasil)")

# Unificar "Portuguese" e "Brazilian Portuguese" como o mesmo idioma
df['supported languages'] = df['supported languages'].apply(lambda x: 'Portuguese' if 'Portuguese' in x else x)

# Filtrando jogos que suportam o idioma Português ou Brasil
df_portuguese = df[df['supported languages'].str.contains('Portuguese', case=False, na=False)]

# Contagem de jogos por idioma
language_count = df_portuguese['supported languages'].value_counts().reset_index()
language_count.columns = ['Idioma', 'Quantidade de Jogos']

# Gráfico de barras interativo
fig_lang = px.bar(language_count, x='Idioma', y='Quantidade de Jogos', 
                  title="Contagem de Jogos por Idioma Suportado (Português/Brasil)",
                  labels={'Idioma': 'Idioma', 'Quantidade de Jogos': 'Quantidade de Jogos'},
                  color='Idioma', 
                  color_discrete_map={'Portuguese': 'purple'})

st.plotly_chart(fig_lang)


# # Filtros de usuário
# st.sidebar.header("Opções de Filtragem")
# genero = st.sidebar.multiselect("Selecione os Gêneros", options=df['Genres'].dropna().unique())
# preco_max = st.sidebar.slider("Preço Máximo", 0, int(df['Price'].max()), 50)
# pontuacao_min = st.sidebar.slider("Pontuação Mínima dos Usuários", 0, int(df['User score'].max()), 70)
# windows = st.sidebar.checkbox("Compatível com Windows", value=True)
# mac = st.sidebar.checkbox("Compatível com Mac", value=False)
# linux = st.sidebar.checkbox("Compatível com Linux", value=False)

# # Converte valores NaN na coluna 'Genres' para strings vazias
# df['Genres'] = df['Genres'].fillna("")

# # Aplicar filtros
# df_filtrado = df.copy()
# if genero:
#     df_filtrado = df_filtrado[df_filtrado['Genres'].apply(lambda x: any(g in x for g in genero))]
# df_filtrado = df_filtrado[(df_filtrado['Price'] <= preco_max) & 
#                           (df_filtrado['User score'] >= pontuacao_min) & 
#                           (df_filtrado['Windows'] == windows) & 
#                           (df_filtrado['Mac'] == mac) & 
#                           (df_filtrado['Linux'] == linux)]

# # Exibir dados filtrados
# st.write("Total de Jogos Encontrados:", len(df_filtrado))
# st.dataframe(df_filtrado[['Name', 'Release date', 'Price', 'User score', 'Genres']])

# # Gráficos de análise
# st.subheader("Análises Gráficas")

# # Gráfico 1: Distribuição de Preços
# st.write("Distribuição de Preços dos Jogos")
# fig1 = px.histogram(df_filtrado, x='Price', nbins=20, title="Distribuição de Preços dos Jogos", 
#                     labels={'Price': 'Preço'}, marginal="box", color_discrete_sequence=['skyblue'])
# fig1.update_layout(bargap=0.1)
# st.plotly_chart(fig1)

# # Gráfico 2: Distribuição de Pontuação dos Usuários
# st.write("Distribuição das Pontuações dos Usuários")
# fig2 = px.histogram(df_filtrado, x='User score', nbins=20, title="Distribuição das Pontuações dos Usuários", 
#                     labels={'User score': 'Pontuação dos Usuários'}, marginal="box", color_discrete_sequence=['purple'])
# st.plotly_chart(fig2)

# # Gráfico 3: Relação entre Preço e Pontuação dos Usuários
# st.write("Relação entre Preço e Pontuação dos Usuários")
# fig3 = px.scatter(df_filtrado, x='Price', y='User score', color='Genres', size='Price', 
#                   title="Relação entre Preço e Pontuação dos Usuários", 
#                   labels={'Price': 'Preço', 'User score': 'Pontuação dos Usuários'})
# st.plotly_chart(fig3)

# # Gráfico 4: Lançamento de Jogos ao Longo do Tempo
# st.write("Lançamento de Jogos ao Longo do Tempo")
# df_lancamento = df_filtrado.groupby(df_filtrado['Release date'].dt.year).size().reset_index(name='Quantidade')
# fig4 = px.area(df_lancamento, x='Release date', y='Quantidade', title="Lançamento de Jogos ao Longo do Tempo", 
#                labels={'Release date': 'Ano', 'Quantidade': 'Quantidade de Jogos'}, color_discrete_sequence=['orange'])
# st.plotly_chart(fig4)

# # Gráfico 5: Gêneros Mais Populares
# st.write("Gêneros Mais Populares")
# generos_populares = df_filtrado['Genres'].value_counts().head(10).reset_index()
# generos_populares.columns = ['Gênero', 'Quantidade']
# fig5 = px.bar(generos_populares, x='Quantidade', y='Gênero', orientation='h', 
#               title="Top 10 Gêneros Mais Populares", 
#               labels={'Quantidade': 'Quantidade de Jogos', 'Gênero': 'Gêneros'}, color='Quantidade', 
#               color_continuous_scale='viridis')
# st.plotly_chart(fig5)
