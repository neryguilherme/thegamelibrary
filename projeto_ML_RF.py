import pandas as pd
import random

# Carregar o arquivo Parquet
data = pd.read_parquet('games_cleaned.parquet')

# Preencher valores nulos
data['Genres'] = data['Genres'].fillna('Unknown')
data['Tags'] = data['Tags'].fillna('Unknown')
data['Categories'] = data['Categories'].fillna('Unknown')

# Filtrar apenas as colunas necessárias
data_filtered = data[['Name', 'Genres', 'Tags', 'Categories']]

# Selecionar 5 jogos aleatórios com gêneros diferentes
unique_genres = data_filtered['Genres'].unique()
selected_games = []

while len(selected_games) < 5 and len(unique_genres) > 0:
    genre = random.choice(unique_genres)
    genre_games = data_filtered[data_filtered['Genres'].str.contains(genre, case=False, na=False)]
    if not genre_games.empty:
        selected_game = genre_games.sample(1)
        selected_games.append(selected_game.iloc[0])
    unique_genres = unique_genres[unique_genres != genre]

print("Selecione um dos seguintes jogos para ver recomendações semelhantes:")
for idx, game in enumerate(selected_games):
    print(f"{idx + 1}. {game['Name']}")

# Obter escolha do usuário
choice = int(input("Digite o número do jogo escolhido (1-5): ")) - 1
chosen_game = selected_games[choice]

# Encontrar jogos semelhantes com base em termos parciais nos gêneros e tags
similar_games = data_filtered[
    (data_filtered['Genres'].str.contains(chosen_game['Genres'].split(',')[0], case=False, na=False)) |
    (data_filtered['Tags'].str.contains(chosen_game['Tags'].split(',')[0], case=False, na=False))
]

# Excluir o jogo escolhido
similar_games = similar_games[similar_games['Name'] != chosen_game['Name']]

# Selecionar até 5 jogos mais semelhantes
final_recommendations = similar_games.head(5)

# Exibir os resultados
if not final_recommendations.empty:
    print("\nRecomendamos os seguintes jogos semelhantes:")
    for game in final_recommendations['Name']:
        print(game)
else:
    print("Nenhum jogo semelhante encontrado com base nos critérios.")