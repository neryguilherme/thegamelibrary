import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from random import sample
import os


def load_data():
    caminho_parquet = os.path.join( os.getcwd(), 'games.parquet')
    
    if not os.path.exists(caminho_parquet):
        print("Arquivo games.parquet não encontrado no diretório atual.")
        return pd.DataFrame()

    # Colunas necessárias para análise
    colunas_necessarias = [
        "Name", "Release date", "Price", "User score", "Genres", 
        "Windows", "Mac", "Linux", 'Tags', 'Categories', 'Recommendations'
    ]
    
    # Carregar apenas as colunas necessárias
    df = pd.read_parquet(caminho_parquet, columns=colunas_necessarias)
    
    # Pré-processamento de dados
    df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
    df['Price'] = df['Price'].fillna(0)
    df['User score'] = df['User score'].fillna(0)
    df['Genres'] = df['Genres'].fillna("Sem Genero")
    
    # Remover linhas duplicadas baseando-se na coluna 'Name' e mantendo somente a primeira ocorrência
    df = df.drop_duplicates(subset=['Name'], keep='first')
    
    return df


def fill_empty_val(orig_data: pd.DataFrame) -> pd.DataFrame:
    # Preencher valores ausentes nas colunas relevantes com strings vazias
    orig_data['Genres'] = orig_data['Genres'].fillna("")
    orig_data['Tags'] = orig_data['Tags'].fillna("")
    orig_data['Categories'] = orig_data['Categories'].fillna("")
    orig_data['Combined_Features'] = orig_data['Genres'] + " " + orig_data['Tags']
    orig_data['Combined_Features'] = orig_data['Combined_Features'].fillna("")

    # Preencher valores ausentes na coluna Recommendations com 0
    orig_data['Recommendations'] = orig_data['Recommendations'].fillna(0)
    return orig_data

def ask_preferences() -> tuple[str]:
    platform_preference = input("Qual plataforma você prefere? (Windows, Linux, Mac): ").capitalize()
    category_preference = input("Você prefere jogos Single-Player ou Multi-Player? ").capitalize()
    genre_preference = input("Digite um gênero que você gosta (ou deixe em branco para ignorar): ").lower()
    tag_preference = input("Digite uma tag que você gosta (ou deixe em branco para ignorar): ").lower()

    return platform_preference, category_preference, genre_preference, tag_preference

def pref_recomm(data: pd.DataFrame, plat_pref: str, categ_pref: str, genre_pref: str, tag_pref: str) -> None:
    # Filtrar o dataset com base nas preferências do usuário
    filtered_data = data[
        (data[plat_pref] == True) &  # noqa: E712
        (data['Categories'].str.contains(categ_pref, case=False)) &
        (data['Genres'].str.contains(genre_pref, case=False) if genre_pref else True) &
        (data['Tags'].str.contains(tag_pref, case=False) if tag_pref else True)
    ]

    # Caso não haja resultados, mostrar mensagem
    if filtered_data.empty:
        print("Nenhum jogo corresponde às suas preferências.")
    else:
        # Aplicar TF-IDF no conjunto filtrado
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(filtered_data['Combined_Features'])

        # Similaridade de cosseno (exemplo: comparando todos os jogos entre si)
        #cosine_sim = cosine_similarity(tfidf_matrix)

        # Ordenar as recomendações por popularidade (Recomendations)
        recommendations = filtered_data.sort_values(by='Recommendations', ascending=False).head(10)
        recomm_list = []
        
        # Exibir apenas os nomes dos jogos recomendados
        print("\nJogos recomendados:")
        for index, row in recommendations.iterrows():
            print(row['Name'])
            recomm_list.append(row['Name'])

    return recomm_list

#Funcoes para recomendacoes baseadas em um jogo escolhido
def show_rand_games(data: pd.DataFrame) -> list[object]:
    # Criar um conjunto de 5 jogos aleatórios de gêneros diferentes
    unique_genres = data['Genres'].str.split(',').explode().unique()  # Lista de gêneros únicos
    random_genres = sample(list(unique_genres), 5)  # Selecionar 5 gêneros aleatórios

    # Selecionar um jogo aleatório para cada gênero
    random_games = []
    for genre in random_genres:
        genre_games = data[data['Genres'].str.contains(genre, case=False)]
        if not genre_games.empty:
            random_games.append(genre_games.sample(1).iloc[0])  # Escolher um jogo aleatório
    
    return random_games

def game_input(rand_games: list[str])-> object:
    # Mostrar os jogos para o usuário escolher
    print("Escolha um jogo dentre as opções abaixo:")
    for i, game in enumerate(rand_games):
        print(f"{i + 1}. {game['Name']} - Gênero: {game['Genres']}")
    
    # Entrada do usuário
    choice = int(input("Digite o número do jogo que você escolhe: "))
    
    """ while not choice.isnumeric():
        global dataset
        show_rand_games(dataset) """
        
    r_choice = choice - 1
    chosen_game = rand_games[r_choice]

    return chosen_game

def recom_five_games(data: pd.DataFrame, chosen_game: object)-> list[str]:
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
    list_recom = []
    for i in range(1, len(similar_games)):  # Começa em 1 para evitar recomendar o próprio jogo
        game_index = similar_games[i][0]
        recommended_game = data.iloc[game_index]
        print(f"{recommended_game['Name']} - Gênero: {recommended_game['Genres']}")
        list_recom.append(f"{recommended_game['Name']}")
        count += 1
        if count == 5:
            break
    
    return list_recom

def main():
    dataset = load_data()
    data = fill_empty_val(dataset)
    user_pref = ask_preferences()
    uplat_pref, ucat_pref, ugenre_pref, utag_pref = user_pref[0], user_pref[1], user_pref[2], user_pref[3]
    pref_recomm(data, uplat_pref, ucat_pref, ugenre_pref, utag_pref)


if __name__ == "__main__":
    main()