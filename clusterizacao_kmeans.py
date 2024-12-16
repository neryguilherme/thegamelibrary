import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


def load_data(file_path):
    """Carrega os dados do arquivo parquet."""
    return pd.read_parquet(file_path)


def preprocess_data(df, features):
    """Seleciona, trata valores ausentes e padroniza os dados."""
    X = df[features].fillna(0)  # Substitui valores ausentes por 0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def find_optimal_clusters(X, max_k=10):
    """
    Determina o número ideal de clusters usando o método Elbow.
    Retorna as inércias e plota o gráfico.
    """
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    # Plotando o método Elbow
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Inércia")
    plt.title("Método Elbow para Determinar o Número de Clusters")
    plt.grid()
    plt.show()

    return inertia


def perform_clustering(X, n_clusters):
    """Aplica o algoritmo KMeans e retorna o modelo treinado e os rótulos."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)
    return kmeans, labels


def analyze_clusters(df, labels, features):
    """Adiciona rótulos ao dataframe e exibe características dos clusters."""
    df['cluster'] = labels

    print("Clusters atribuídos:")
    print(df[['Name', 'cluster']].head())

    # Exibe estatísticas de cada cluster
    for i in range(df['cluster'].nunique()):
        print(f"\nCaracterísticas do Cluster {i}:")
        print(df[df['cluster'] == i][features].mean())


def main():
    # Caminho do arquivo e seleção de features

    FILE_PATH = os.path.join(os.getcwd(), "games_cleaned.parquet")
    FEATURES = ['Positive', 'Negative', 'Price', 'Recommendations']

    # Carrega e processa os dados
    df = load_data(FILE_PATH)
    X_scaled, _ = preprocess_data(df, FEATURES)

    # Determina o número ótimo de clusters
    find_optimal_clusters(X_scaled, max_k=10)
    optimal_k = 3  # Defina com base no método Elbow

    # Executa o KMeans e analisa os clusters
    _, labels = perform_clustering(X_scaled, n_clusters=optimal_k)
    analyze_clusters(df, labels, FEATURES)


if __name__ == "__main__":
    main()
