import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carregar o dataset
data = pd.read_csv('games.csv', encoding='utf-8')

# Preencher valores ausentes nas colunas relevantes com strings vazias
data['Genres'] = data['Genres'].fillna("")
data['Tags'] = data['Tags'].fillna("")
data['Categories'] = data['Categories'].fillna("")
data['Combined_Features'] = data['Genres'] + " " + data['Tags']
data['Combined_Features'] = data['Combined_Features'].fillna("")

# Preencher valores ausentes na coluna Recommendations com 0
data['Recommendations'] = data['Recommendations'].fillna(0)

# Perguntar preferências do usuário
platform_preference = input("Qual plataforma você prefere? (Windows, Linux, Mac): ").capitalize()
category_preference = input("Você prefere jogos Single-Player ou Multi-Player? ").capitalize()
genre_preference = input("Digite um gênero que você gosta (ou deixe em branco para ignorar): ").lower()
tag_preference = input("Digite uma tag que você gosta (ou deixe em branco para ignorar): ").lower()

# Filtrar o dataset com base nas preferências do usuário
filtered_data = data[
    (data[platform_preference] == True) &
    (data['Categories'].str.contains(category_preference, case=False)) &
    (data['Genres'].str.contains(genre_preference, case=False) if genre_preference else True) &
    (data['Tags'].str.contains(tag_preference, case=False) if tag_preference else True)
]

# Caso não haja resultados, mostrar mensagem
if filtered_data.empty:
    print("Nenhum jogo corresponde às suas preferências.")
else:
    # Aplicar TF-IDF no conjunto filtrado
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(filtered_data['Combined_Features'])

    # Similaridade de cosseno (exemplo: comparando todos os jogos entre si)
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Ordenar as recomendações por popularidade (Recomendations)
    recommendations = filtered_data.sort_values(by='Recommendations', ascending=False).head(10)

    # Exibir apenas os nomes dos jogos recomendados
    print("\nJogos recomendados:")
    for index, row in recommendations.iterrows():
        print(row['Name'])