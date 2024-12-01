import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


dataset = pd.read_csv('games.csv', encoding='utf-8')

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

def give_recommendation(data: pd.DataFrame, plat_pref: str, categ_pref: str, genre_pref: str, tag_pref: str) -> None:
    # Filtrar o dataset com base nas preferências do usuário
    filtered_data = data[
        (data[plat_pref] == True) &  
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

        # Exibir apenas os nomes dos jogos recomendados
        print("\nJogos recomendados:")
        for index, row in recommendations.iterrows():
            print(row['Name'])
    return

def main():
    global dataset
    data = fill_empty_val(dataset)
    user_pref = ask_preferences()
    uplat_pref, ucat_pref, ugenre_pref, utag_pref = user_pref[0], user_pref[1], user_pref[2], user_pref[3]
    give_recommendation(data, uplat_pref, ucat_pref, ugenre_pref, utag_pref)


main()