import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime
from collections import Counter
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path, columns=["Price", "Positive", "Release date", "Tags"])

# Calcular a idade do jogo
def idade_jogo(release_date):
    try:
        release_date = datetime.strptime(release_date, "%b %d, %Y")
        return datetime.now().year - release_date.year
    except:
        return np.nan

# Método do cotovelo
def elbow_method(data, max_k=10):
    distortions = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    return distortions

# Método da silhueta
def silhouette_method(data, max_k=10):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))
    return silhouette_scores

# K-Means
def run_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data["Cluster"] = kmeans.fit_predict(df_scaled)
    return data, kmeans

st.title("Clusterização de Jogos (Preço x Avaliações Positivas)")

file_path = "games_cleaned.parquet"

data = load_data(file_path)

# Remover valores nulos
data.dropna(inplace=True)

# Criar a coluna de idade do jogo
data["Game Age"] = data["Release date"].apply(idade_jogo)

data.dropna(inplace=True)  # Remover linhas com valores NaN após a conversão da data

# Padronizar os dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(data[["Price", "Positive"]])

# Escolher o número máximo de clusters para análise
max_k = st.sidebar.slider("Número máximo de clusters para avaliação:", min_value=5, max_value=15, value=10)

# Escolha do número de clusters
n_clusters = st.sidebar.slider("Escolha o número de clusters (K):", min_value=2, max_value=max_k, value=3)
clustered_data, kmeans_model = run_kmeans(data, n_clusters)

# Visualização dos Clusters
st.subheader("Visualização dos Clusters")
fig, ax = plt.subplots()
sns.scatterplot(x=clustered_data["Price"], y=clustered_data["Positive"], hue=clustered_data["Cluster"], palette="viridis", ax=ax)
ax.set_xlabel("Preço")
ax.set_ylabel("Avaliações Positivas")
ax.set_title("Clusters de Jogos")
st.pyplot(fig)

# Método da Silhueta
st.subheader("Índice de Silhueta")
silhouette_scores = silhouette_method(df_scaled, max_k)
fig, ax = plt.subplots()
ax.plot(range(2, max_k + 1), silhouette_scores, marker='o', color='green')
ax.set_xlabel("Número de Clusters (K)")
ax.set_ylabel("Índice de Silhueta")
ax.set_title("Método do Cotovelo")
st.pyplot(fig)

# Exibir o índice de silhueta para o número escolhido de clusters
silhouette_avg = silhouette_score(df_scaled, clustered_data["Cluster"])
st.subheader("Qualidade do Cluster")
st.write(f"Índice de Silhueta para K={n_clusters}: *{silhouette_avg:.2f}*")

# Idade dos jogos por cluster
st.subheader("Distribuição da Idade dos Jogos por Cluster")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=clustered_data, x="Cluster", y="Game Age", palette="viridis", ax=ax)
ax.set_xlabel("Cluster")
ax.set_ylabel("Idade do Jogo (anos)")
ax.set_title("Distribuição da Idade dos Jogos por Cluster")
st.pyplot(fig)

# 3 tags mais populares em cada cluster
def get_top_tags_per_cluster(data):
    top_tags = {}
    for cluster in sorted(data["Cluster"].unique()):
        tags = data[data["Cluster"] == cluster]["Tags"].dropna()
        all_tags = [tag for sublist in tags.str.split(",") for tag in sublist]
        common_tags = [tag for tag, _ in Counter(all_tags).most_common(3)]
        top_tags[cluster] = common_tags
    return top_tags

clustered_data, kmeans_model = run_kmeans(data, n_clusters)

# Contar a quantidade de jogos por cluster
cluster_counts = clustered_data["Cluster"].value_counts().sort_index()

# Exibir a quantidade de jogos por cluster no Streamlit
st.subheader("Quantidade de Jogos por Cluster")
for cluster, count in cluster_counts.items():
    st.write(f"Cluster {cluster}: {count} jogos")

# Exibir as 3 tags mais populares de cada cluster
top_tags_per_cluster = get_top_tags_per_cluster(clustered_data)
st.subheader("Tags Mais Populares por Cluster")
for cluster, tags in top_tags_per_cluster.items():
    st.write(f"Cluster {cluster}: {', '.join(tags)}")

# Exibir o gráfico de análise de silhueta
def plot_silhouette_analysis(data, labels, n_clusters):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Gráfico de silhueta
    silhouette_vals = silhouette_samples(data, labels)
    y_lower, y_upper = 0, 0
    for i in range(n_clusters):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, alpha=0.7, color=cm.nipy_spectral(float(i) / n_clusters))
        y_lower = y_upper

    ax.axvline(x=silhouette_score(data, labels), color="red", linestyle="--")
    ax.set_xlabel("Coeficiente de Silhueta")
    ax.set_ylabel("Cluster")
    ax.set_title("Gráfico de Silhueta")

    st.pyplot(fig)

# Gerar e exibir o gráfico de silhueta
plot_silhouette_analysis(df_scaled, clustered_data["Cluster"].values, n_clusters)