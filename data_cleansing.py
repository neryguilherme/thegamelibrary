import pandas as pd
import os

def load_data():
    """Carrega o dataset Parquet."""
    path_parquet = os.path.join(os.getcwd(), "games.parquet")
    
    if not os.path.exists(path_parquet):
        print("Arquivo games.parquet não encontrado no diretório atual.")
        return pd.DataFrame()

    return pd.read_parquet(path_parquet)

def clean_data(df: pd.DataFrame):
    """Realiza o processo de limpeza de dados."""

    # Remover as colunas que nao sao utilizadas 
    l_columns = ['AppID', 'Peak CCU', 'Required age', 'DLC count', 'About the game', 'Reviews', 'Website', 'Support url',
                 'Support email', 'Metacritic score', 'Metacritic url', 'Score rank', 'Notes', 'Developers', 'Publishers', 
                 'Screenshots','Movies']
    
    df = df.drop(columns= l_columns, axis=1)

    # Remover duplicatas
    df = df.drop_duplicates(subset=["Name"], keep='first')

    # Preencher valores numéricos ausentes com 0
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].fillna(0)

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].fillna(0.0)

    # Verificar tipos de dados e corrigir se necessário
    # Transformar coluna de data para datetime
    df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')  
    df['Genres'] = df['Genres'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

    # Filtrar dados inválidos (exemplo: valores negativos em colunas específicas)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[df[col] < 0] = 0

    return df

def save_data(df: pd.DataFrame):
    """Salva o dataset limpo em um novo arquivo Parquet."""
    output_path = os.path.join(os.getcwd(), "games_cleaned.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Dados limpos salvos em: {output_path}")

def main():
    df = load_data()

    if df.empty:
        return

    df_cleaned = clean_data(df)

    save_data(df_cleaned)

if __name__ == "__main__":
    main()
