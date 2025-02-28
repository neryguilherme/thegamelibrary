import pandas as pd
import streamlit as st
import re 
import sys
import prediction as pred
import matplotlib.pyplot as plt
from collections import Counter

# Carregar o dataset Parquet
FILE_PATH = "games.parquet"

# Função para carregar os dados com cache
@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path)

df = load_data(FILE_PATH)

# Obter valores únicos da coluna "Supported languages"
supported_languages = []
invalid_chars = re.compile(r"[;/\\_&\[\]]")  
for languages in df["Supported languages"].dropna():
    languages = languages.strip("[]")  
    found_languages = re.findall(r"'([^']+)'", languages)  # Encontrar valores entre aspas simples
    for lang in found_languages:
        if not invalid_chars.search(lang) and lang not in supported_languages:
            supported_languages.append(lang)

# Obter valores únicos da coluna "Categories"
unique_categories = []
for categories in df['Categories'].dropna():
    list_categories = categories.split(",")
    for category in list_categories:
        if category not in unique_categories:
            unique_categories.append(category)

# Obter valores únicos da coluna "Categories"
unique_tags = []
for tags in df['Tags'].dropna():
    list_tags = tags.split(",")
    for tag in list_tags:
        if tag not in unique_tags:
            unique_tags.append(tag)

# Obter faixa de preços da coluna "Price"
min_price = df["Price"].min()
max_price = df["Price"].max()
st.title("Selecione a faixa de preço em Dolares")
price_range = st.slider("Faixa de Preço", float(min_price), float(max_price), (float(min_price), float(max_price)))

# Criar a interface com Streamlit
st.title("Selecione uma Tag")

# Criar um dropdown
selected_tag = st.selectbox("Tags", unique_tags)

st.write(f"Você selecionou: {selected_tag}")

# Criar a interface com Streamlit
st.title("Selecione uma Categoria")

# Criar um dropdown
selected_category = st.selectbox("Categorias", unique_categories)

st.write(f"Você selecionou: {selected_category}")

# Criar a interface com Streamlit
st.title("Selecione um idioma suportado")

# Criar um dropdown
selected_language = st.selectbox("Idioma suportado", supported_languages)

st.write(f"Você selecionou: {selected_language}")

if st.button("Aplicar Filtros"):
    # Lista de categorias a serem contadas
    categorias = ['Indie', 'Action', 'Casual', 'Outros']

    # Lista de previsões (substituir por valores reais)
    previsao = pred.prediction(price_range[0], price_range[1], selected_tag, selected_category, selected_language)

    # Contar as ocorrências de cada categoria
    contagem = Counter(previsao)

    total = sum(contagem.values())

    # Garantir que todas as categorias tenham um valor na contagem e calcular porcentagem
    contagem_final = {categoria: (contagem.get(categoria, 0) / total) * 100 for categoria in categorias}

    # Criar o gráfico de barras com eixos invertidos
    fig, ax = plt.subplots()
    ax.barh(list(contagem_final.keys()), list(contagem_final.values()), color=['blue', 'red', 'green', 'gray'])
    ax.set_ylabel('Categorias')
    ax.set_xlabel('Porcentagem (%)')
    ax.set_title('Frequência de Categorias na Previsão (Porcentagem)')

    # Exibir gráfico no Streamlit
    st.pyplot(fig)
