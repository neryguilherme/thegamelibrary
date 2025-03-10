import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import shap
import streamlit as st

# Carregar o dataset
data = pd.read_parquet('games_cleaned.parquet')

# Preencher valores ausentes
data['Genres'] = data['Genres'].fillna("")
data['Tags'] = data['Tags'].fillna("")
data['Categories'] = data['Categories'].fillna("")

# Criar coluna Target fictícia
if 'Target' not in data.columns:
    st.warning("'Target' não encontrado no dataset. Criando coluna fictícia para teste.")
    data['Target'] = (data.index % 2)

# Amostra aleatória de 100 jogos
data_sampled = data.sample(100, random_state=42)

# Combinar features de gêneros e tags
data_sampled['Combined_Features'] = data_sampled['Genres'] + " " + data_sampled['Tags']
data_sampled = data_sampled[['Combined_Features', 'Target']]

# Separar features e alvo
X = data_sampled['Combined_Features']
y = data_sampled['Target']

# Transformar texto em representações numéricas com TF-IDF
tfidf = TfidfVectorizer(max_features=1000, min_df=2, stop_words='english')
X_transformed = tfidf.fit_transform(X)

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Explicar modelo com SHAP
explainer = shap.KernelExplainer(model.predict_proba, X_train, link="logit")
shap_values = explainer.shap_values(X_test, nsamples=100)

# Criar DataFrame com nomes das features
feature_names = tfidf.get_feature_names_out()
X_test_df = pd.DataFrame(X_test.toarray(), columns=feature_names)

# Exibir gráficos no Streamlit
st.title("Análise de Importância com SHAP e TF-IDF")

# Mostrar gráfico de importância das features
if len(model.classes_) > 2:
    st.subheader("Modelo Multiclasse: Importância Combinada")
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
else:
    st.subheader("Modelo Binário: Importância de Features")
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)

st.pyplot()
