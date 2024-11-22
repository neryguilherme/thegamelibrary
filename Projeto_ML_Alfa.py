import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carregar o dataset
data = pd.read_csv('games_dataset.csv')

# Combinar 'genres' e 'tags' em uma única coluna para análise
data['combined_features'] = data['genres'] + " " + data['tags']

# Vetorização usando TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combined_features'])

# Similaridade de cosseno
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Normalizar a coluna 'recomendations' entre 0 e 1
data['normalized_recommendations'] = data['recomendations'] / data['recomendations'].max()

# Função para recomendar jogos com base nas respostas do usuário
def recommend_based_on_user_input(data, similarity_matrix, top_n=5):
    # Perguntas ao usuário
    print("Bem-vindo ao sistema de recomendação de jogos!")
    preferred_genre = input("Qual gênero de jogo você prefere (ex.: RPG, Ação, Plataforma)? ").lower()
    preferred_tag = input("Que tipo de característica você busca em um jogo (ex.: Multiplayer, Retro, Relaxante)? ").lower()
    
    # Filtrar jogos que correspondem às preferências
    filtered_data = data[
        (data['genres'].str.lower().str.contains(preferred_genre, na=False)) &
        (data['tags'].str.lower().str.contains(preferred_tag, na=False))
    ]
    
    # Caso nenhum jogo seja encontrado com as preferências
    if filtered_data.empty:
        print("Nenhum jogo encontrado com essas preferências. Recomendando jogos populares...")
        filtered_data = data  # Considera todos os jogos se o filtro não encontrar nenhum
    
    # Escolher um jogo inicial para calcular similaridade
    base_game_idx = filtered_data.index[0]
    
    # Calcular o peso de'recomendations'
    similarity_scores = list(enumerate(similarity_matrix[base_game_idx]))
    adjusted_scores = [
        (idx, score * data.iloc[idx]['normalized_recommendations'])
        for idx, score in similarity_scores
    ]
    
    # Ordenar
    sorted_scores = sorted(adjusted_scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Obter os nomes
    recommended_games = [data.iloc[i[0]]['name'] for i in sorted_scores]
    
    print("\nJogos recomendados para você:")
    for game in recommended_games:
        print(f"- {game}")

recommend_based_on_user_input(data, similarity_matrix, top_n=5)