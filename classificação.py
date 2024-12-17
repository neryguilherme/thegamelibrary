import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Passo 1: Carregar o dataset

def classificacao(df):
# Passo 2: Selecionar colunas relevantes
# Vamos usar colunas que sejam numéricas ou facilmente processadas para o modelo
 colunas_selecionadas = [
    'Peak CCU', 'Required age', 'Price', 'DLC count', 'Metacritic score',
    'User score', 'Positive', 'Negative', 'Achievements', 
    'Average playtime forever', 'Median playtime forever'
]

# Filtrar o dataset com as colunas selecionadas e remover valores ausentes
 df = df[colunas_selecionadas + ['Recommendations']].dropna()

# Converter a coluna de "Recommendations" em binária: 1 para recomendado, 0 caso contrário
 df['Recommendations'] = (df['Recommendations'] > 0).astype(int)

# Passo 3: Dividir os dados em treino e teste
 X = df[colunas_selecionadas]
 y = df['Recommendations']

 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados numéricos para o KNN
 scaler = StandardScaler()
 X_train_scaled = scaler.fit_transform(X_train)
 X_test_scaled = scaler.transform(X_test)

# Passo 4: Treinar e avaliar o modelo KNN
 knn = KNeighborsClassifier(n_neighbors=5)
 knn.fit(X_train_scaled, y_train)
 y_pred_knn = knn.predict(X_test_scaled)

 print("Resultados do KNN:")
 print("Acurácia:", accuracy_score(y_test, y_pred_knn))
 print(classification_report(y_test, y_pred_knn))

# Passo 5: Treinar e avaliar o modelo Random Forest
 rf = RandomForestClassifier(n_estimators=100, random_state=42)
 rf.fit(X_train, y_train)
 y_pred_rf = rf.predict(X_test)

 print("\nResultados do Random Forest:")
 print("Acurácia:", accuracy_score(y_test, y_pred_rf))
 print(classification_report(y_test, y_pred_rf))



def main ():
 df = pd.read_parquet("games.parquet")
 classificacao(df)


if __name__ == "__main__":
    main()
