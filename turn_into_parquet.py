import pandas as pd
df = pd.read_csv('games.csv')
df.to_parquet('games.parquet')