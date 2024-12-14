import pandas as pd
import os

def load_data():
    """Carrega o dataset Parquet."""
    path_parquet = os.path.join(os.getcwd(), "games.parquet")

    if not os.path.exists(path_parquet):
        print("Arquivo games.parquet não encontrado no diretório atual.")
        return pd.DataFrame()

    return pd.read_parquet(path_parquet)

def clean_data(df):
    """Realiza o processo de limpeza de dados."""
    # 1. Remover duplicatas
    df = df.drop_duplicates()

    # 2. Identificar e tratar valores ausentes
    # Exemplo: preencher valores numéricos ausentes com a média da coluna
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].mean())

    # Preencher valores categóricos ausentes com 'Desconhecido'
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Desconhecido')

    # 3. Verificar tipos de dados e corrigir se necessário
    # Exemplo: transformar colunas de data para datetime
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except (ValueError, TypeError):
            pass  # Ignorar colunas que não são datas

    # 4. Filtrar dados inválidos (exemplo: valores negativos em colunas específicas)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df = df[df[col] >= 0]

    return df

def save_data(df):
    """Salva o dataset limpo em um novo arquivo Parquet."""
    output_path = os.path.join(os.getcwd(), "games_cleaned.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Dados limpos salvos em: {output_path}")

def main():
    # Carregar dados
    df = load_data()

    if df.empty:
        return

    # Limpar dados
    df_cleaned = clean_data(df)

    # Salvar dados limpos
    save_data(df_cleaned)

if __name__ == "__main__":
    main()
