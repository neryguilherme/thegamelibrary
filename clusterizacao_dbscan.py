import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os


def load_data(file_path):
    """Carrega os dados do arquivo Parquet."""
    return pd.read_parquet(file_path)


def preprocess_data(df, features):
    """Seleciona as features, lida com valores ausentes e padroniza os dados."""
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def perform_dbscan(X, eps=0.5, min_samples=5):
    """Executa o algoritmo DBSCAN e retorna os rótulos dos clusters."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(X)


def analyze_clusters(df, features):
    """Analisa e imprime as estatísticas dos clusters."""
    clusters = df['cluster'].unique()
    for cluster in clusters:
        if cluster == -1:
            print("Pontos de Ruído:")
        else:
            print(f"Cluster {cluster}:")
        print(df[df['cluster'] == cluster][features].mean())


def main():
    # Caminho do arquivo e seleção das features

    FILE_PATH = os.path.join(os.getcwd(), "games_cleaned.parquet")
    FEATURES = ['Positive', 'Negative', 'Price', 'Recommendations']

    # Carregar e preparar os dados
    df = load_data(FILE_PATH)
    X_scaled = preprocess_data(df, FEATURES)

    # Realizar DBSCAN
    EPS = 0.5  # Ajuste de acordo com a necessidade
    MIN_SAMPLES = 5
    df['cluster'] = perform_dbscan(X_scaled, eps=EPS, min_samples=MIN_SAMPLES)

    # Exibir os clusters
    print(df[['Name', 'cluster']])

    # Analisar clusters
    analyze_clusters(df, FEATURES)


if __name__ == "__main__":
    main()
