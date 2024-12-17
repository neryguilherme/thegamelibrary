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

    l_columns = ['AppID', 'Peak CCU', 'DLC count', 'About the game', 'Reviews', 'Website', 'Support url',
                 'Support email', 'Metacritic score', 'Metacritic url', 'Score rank', 'Notes', 'Developers', 'Publishers', 
                 'Screenshots','Movies']
    
    df = df.drop(columns= l_columns, axis=1)

    df = df.drop_duplicates(subset=["Name"], keep='first')

    # Preencher valores numéricos ausentes com 0
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].fillna(0)

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].fillna(0.0)

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
