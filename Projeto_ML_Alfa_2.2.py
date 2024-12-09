import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Carregar o dataset
data = pd.read_parquet('games.parquet', encoding='utf-8')

# Preencher valores ausentes
data['Genres'] = data['Genres'].fillna("")
data['Tags'] = data['Tags'].fillna("")
data['Categories'] = data['Categories'].fillna("")
data['Combined_Features'] = data['Genres'] + " " + data['Tags']
data['Combined_Features'] = data['Combined_Features'].fillna("")
data['Recomendations'] = data['Recomendations'].fillna(0)

# Criar um conjunto de 5 jogos aleatórios de gêneros diferentes
unique_genres = data['Genres'].str.split(',').explode().unique()  # Lista de gêneros únicos
random_genres = random.sample(list(unique_genres), 5)  # Selecionar 5 gêneros aleatórios

# Selecionar um jogo aleatório para cada gênero
random_games = []
for genre in random_genres:
    genre_games = data[data['Genres'].str.contains(genre, case=False)]
    if not genre_games.empty:
        random_games.append(genre_games.sample(1).iloc[0])  # Escolher um jogo aleatório

# Mostrar os jogos para o usuário escolher
print("Escolha um jogo dentre as opções abaixo:")
for i, game in enumerate(random_games):
    print(f"{i + 1}. {game['Name']} - Gênero: {game['Genres']}")

# Entrada do usuário
choice = int(input("Digite o número do jogo que você escolhe: ")) - 1
chosen_game = random_games[choice]

# Calcular similaridade com base no jogo escolhido
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['Combined_Features'])
cosine_sim = cosine_similarity(tfidf_matrix)

chosen_index = data[data['Name'] == chosen_game['Game_Name']].index[0]
similarities = list(enumerate(cosine_sim[chosen_index]))

# Ordenar jogos similares
similar_games = sorted(similarities, key=lambda x: x[1], reverse=True)

# Mostrar os 5 jogos mais similares (excluindo o próprio jogo)
print("\nJogos semelhantes ao escolhido:")
count = 0
for i in range(1, len(similar_games)):  # Começa em 1 para evitar recomendar o próprio jogo
    game_index = similar_games[i][0]
    recommended_game = data.iloc[game_index]
    print(f"{recommended_game['Name']} - Gênero: {recommended_game['Genres']}")
    count += 1
    if count == 5:
        break

def main():
    global data

if __name__ == "__main__":
    main()