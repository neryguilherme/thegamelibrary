import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
import seaborn as sns
from google.colab import files

# Carregar os dados
banco = pd.read_parquet('/content/games_preprocessed.parquet')

# Função otimizada para processar os gêneros
def process_genres(genres_str):
    if not isinstance(genres_str, str):
        return "Outros"
    
    # Remove espaços em branco e usa conjunto para evitar duplicatas
    genres = set(g.strip() for g in genres_str.split(','))
    valid_genres = {"Indie", "Action", "Casual"}
    
    # Mantém apenas os gêneros válidos, ordenados alfabeticamente
    filtered_genres = sorted(valid_genres.intersection(genres))
    # Se houver gêneros que não são válidos, adiciona "Outros"
    if len(filtered_genres) < len(genres):
        filtered_genres.append("Outros")
    
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
    max_depth=15,        # Controla complexidade do modelo
    n_estimators=150,   # Mais estimadores podem melhorar o desempenho
    subsample=0.4,      # Amostragem para reduzir overfitting
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

#Matriz de Confusão com rótulos
plt.figure(figsize=(6,6))
genre_labels = label_encoders['Genres'].inverse_transform(np.unique(y_test)) #Obtém os nomes dos gêneros
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
            xticklabels=genre_labels, yticklabels=genre_labels)
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão XGBoost")
plt.savefig('confusion_matrix_labeled.png') # Salva a figura em um arquivo PNG
plt.show()

# Baixe a imagem para o seu computador local
files.download('confusion_matrix_labeled.png')

explainer = shap.TreeExplainer(xgb_model)
X_test_sample = X_test.sample(random_state=42)
# Calcula os valores SHAP para o conjunto de teste
shap_values = explainer.shap_values(X_test, approximate=True)

# Gráfico summary plot de barras (importância média das features)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig('shap_summary_bar.png')
plt.show()

# Gráfico summary plot padrão (distribuição dos impactos das features)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('shap_summary.png')
plt.show()