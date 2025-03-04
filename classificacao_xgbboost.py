import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
import seaborn as sns

# Carregar os dados
banco = pd.read_parquet('/content/games_preprocessed.parquet')

# Função otimizada para processar os gêneros
def process_genres(genres_str):
    if not isinstance(genres_str, str):
        return "Outros"

    genres = set(genres_str.split(','))  # Usando conjunto para evitar duplicatas
    valid_genres = {"Indie", "Action", "Casual"}
    
    filtered_genres = sorted(valid_genres.intersection(genres))  # Mantém apenas os gêneros desejados
    if len(filtered_genres) < len(genres):
        filtered_genres.append("Outros")  # Se houver outros gêneros, adiciona "Outros"

    return ", ".join(filtered_genres)

banco['Genres'] = banco['Genres'].apply(process_genres)

# Preenchimento de valores ausentes (otimizado)
for col in banco.columns:
    if banco[col].dtype == 'object':
        banco[col].fillna(banco[col].mode()[0], inplace=True)
    else:
        banco[col].fillna(banco[col].median(), inplace=True)

# Aplicação de Label Encoding nos atributos categóricos
label_encoders = {}
x_column = banco.copy()

for column in x_column.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    x_column[column] = le.fit_transform(x_column[column])
    label_encoders[column] = le

# Definição de X (features) e y (target)
X = x_column.drop(columns=['Genres']).copy()
y = x_column['Genres'].copy()  # Mantém separadamente para evitar conflitos

# Escalonamento dos dados
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=['number']).columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo XGBoost com hiperparâmetros ajustados
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y_train)),
    random_state=42,
    learning_rate=0.1,  # Reduzindo taxa de aprendizado para melhor generalização
    max_depth=6,        # Controla complexidade do modelo
    n_estimators=150,   # Mais estimadores podem melhorar o desempenho
    subsample=0.8,      # Amostragem para reduzir overfitting
    colsample_bytree=0.8
)

# Treinamento do modelo
xgb_model.fit(X_train, y_train)

# Previsões
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)



# Métricas
def print_metrics(y_true, y_pred, dataset_name):
    print(f"\nMétricas para {dataset_name}:")
    print(classification_report(y_true, y_pred, zero_division=0.0))
    print(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precisão: {precision_score(y_true, y_pred, average='weighted', zero_division=0.0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted', zero_division=0.0):.4f}")
    print(f"F1-score: {f1_score(y_true, y_pred, average='weighted', zero_division=0.0):.4f}")

print_metrics(y_train, y_pred_train, "Treinamento")
print_metrics(y_test, y_pred_test, "Teste")

# Matriz de Confusão
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cmap='Blues', fmt='d')
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão XGBoost")
plt.show()
# ... (Rest of your code) ...