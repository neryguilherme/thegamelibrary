import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path, columns=["Price", "Positive"])

# Função para calcular o método do cotovelo
def elbow_method(data, max_k=10):
    distortions = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    return distortions

# Função para calcular o método da silhueta
def silhouette_method(data, max_k=10):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))
    return silhouette_scores

# Função para rodar o K-Means
def run_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data["Cluster"] = kmeans.fit_predict(data)
    return data, kmeans

st.title("Clusterização de Jogos (Preço x Avaliações Positivas)")

file_path = "games_cleaned.parquet"

data = load_data(file_path)

# Selecionar apenas 10.000 amostras aleatórias
if len(data) > 10000:
    data = data.sample(n=10000, random_state=42)

# Remover valores nulos
data.dropna(inplace=True)

# Padronizar os dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(data)

st.subheader("Pré-visualização dos Dados")
st.write(data.head())

# Escolher o número máximo de clusters para análise
max_k = st.sidebar.slider("Número máximo de clusters para avaliação:", min_value=5, max_value=15, value=10)

# Método do Cotovelo
st.subheader("Método do Cotovelo")
distortions = elbow_method(df_scaled, max_k)
fig, ax = plt.subplots()
ax.plot(range(2, max_k + 1), distortions, marker='o')
ax.set_xlabel("Número de Clusters (K)")
ax.set_ylabel("Distortion (Inércia)")
ax.set_title("Método do Cotovelo")
st.pyplot(fig)

# Método da Silhueta
st.subheader("Índice de Silhueta")
silhouette_scores = silhouette_method(df_scaled, max_k)
fig, ax = plt.subplots()
ax.plot(range(2, max_k + 1), silhouette_scores, marker='o', color='green')
ax.set_xlabel("Número de Clusters (K)")
ax.set_ylabel("Índice de Silhueta")
ax.set_title("Análise de Silhueta")
st.pyplot(fig)

# Escolha do número de clusters
n_clusters = st.sidebar.slider("Escolha o número de clusters (K):", min_value=2, max_value=max_k, value=3)

# Rodar o K-Means com o valor escolhido
clustered_data, kmeans_model = run_kmeans(data, n_clusters)

# Exibir clusters
st.subheader("Clusters Gerados")
st.write(clustered_data.head())
st.subheader("Visualização dos Clusters")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=clustered_data,
    x="Price",
    y="Positive",
    hue="Cluster",
    palette="viridis",
    ax=ax
)
ax.set_xlabel("Preço")
ax.set_ylabel("Avaliações Positivas")
ax.set_title("Clusters de Jogos (Preço x Avaliações Positivas)")
st.pyplot(fig)

# Exibir o índice de silhueta para o número escolhido de clusters
silhouette_avg = silhouette_score(df_scaled, clustered_data["Cluster"])
st.subheader("Qualidade do Cluster")
st.write(f"Índice de Silhueta para K={n_clusters}: **{silhouette_avg:.2f}**")
