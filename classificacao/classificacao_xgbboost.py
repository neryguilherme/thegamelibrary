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
import pickle
from imblearn.over_sampling import SMOTE  # Importando o SMOTE

# Carregar os dados
banco = pd.read_parquet('/content/games_preprocessed.parquet')

# Função otimizada para processar os gêneros
def process_genres(genres_str):
    if not isinstance(genres_str, str):
        return "Outros"
    genres = set(g.strip() for g in genres_str.split(','))
    valid_genres = {"Indie", "Action", "Casual"}
    filtered_genres = sorted(valid_genres.intersection(genres))
    if len(filtered_genres) < len(genres):
        filtered_genres.append("Outros")
    return ", ".join(filtered_genres)

banco['Genres'] = banco['Genres'].apply(process_genres)

# Preenchimento de valores ausentes
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

for label in label_encoders:
    local_path = f'/content/label_encoders/{label}.pkl'
    with open(local_path, 'wb') as model_file:
        pickle.dump(label_encoders[label], model_file)

# Definição de X (features) e y (target)
X = x_column.drop(columns=['Genres']).copy()
y = x_column['Genres'].copy()

# Aplicando SMOTE para balanceamento das classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Escalonamento dos dados
scaler = StandardScaler()
numerical_cols = X_resampled.select_dtypes(include=['number']).columns
X_resampled[numerical_cols] = scaler.fit_transform(X_resampled[numerical_cols])

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Modelo XGBoost
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y_train)),
    random_state=42,
    learning_rate=0.1,
    max_depth=15,
    n_estimators=150,
    subsample=0.4,
    colsample_bytree=0.8
)

# Treinamento do modelo
xgb_model.fit(X_train, y_train)

local_path = '/content/xgb_model.pkl'
with open(local_path, 'wb') as model_file:
    pickle.dump(xgb_model, model_file)

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
plt.figure(figsize=(6,6))
genre_labels = label_encoders['Genres'].inverse_transform(np.unique(y_test))
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=genre_labels, yticklabels=genre_labels)
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão XGBoost")
plt.savefig('confusion_matrix_labeled.png')
plt.show()

files.download('confusion_matrix_labeled.png')

# SHAP
explainer = shap.TreeExplainer(xgb_model)
X_test_sample = X_test.sample(random_state=42)
shap_values = explainer.shap_values(X_test, approximate=True)

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig('shap_summary_bar.png')
plt.show()

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('shap_summary.png')
plt.show()
