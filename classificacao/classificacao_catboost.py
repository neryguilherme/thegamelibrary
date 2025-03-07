# -*- coding: utf-8 -*-
"""classificacao_CatBoost.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cGWx34aN0Q3Vo2lcIZ-i9EH2uFJzNAQW
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report



"""<a href="https://colab.research.google.com/github/neryguilherme/thegamelibrary/blob/main/Classifica%C3%A7%C3%A3o_Mlp.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>"""

# Carregar os dados
banco = pd.read_parquet('/content/games_preprocessed.parquet')

banco

# prompt:  para cada linha na coluna "Genres" verifique se os valores "Indie", "Action", "Casual" estão presentes, caso existam mantenha eles na lista. Em seguida verifique se há outros valores que NÃO sejam "Indie", "Action", "Casual", caso existam substitua todos por "Outros". Em seguida ordene os valores da célula de acordo com a ordem seguinte: ["Indie","Action", "Casual", "Outros"]

def process_genres(genres_str):
    genres = []
    if isinstance(genres_str, str):
      genres = genres_str.split(',')

    processed_genres = []
    others_present = False

    for genre in genres:
        genre = genre.strip()
        if genre in ["Indie", "Action", "Casual"]:
            processed_genres.append(genre)
        elif genre != "":
            others_present = True

    if others_present:
        processed_genres.append("Outros")

    ordered_genres = []
    for target_genre in ["Indie", "Action", "Casual", "Outros"]:
      if target_genre in processed_genres:
        ordered_genres.append(target_genre)

    return ", ".join(ordered_genres)


banco['Genres'] = banco['Genres'].apply(process_genres)
banco['Genres']

# Preenchimento de valores faltantes
for col in banco.columns:
    if banco[col].dtype == 'object':
        banco[col] = banco[col].fillna(banco[col].mode()[0])  # Preenche com a moda
    elif pd.api.types.is_numeric_dtype(banco[col]):
        banco[col] = banco[col].fillna(banco[col].median())  # Preenche com a mediana
banco

# prompt: aplique o label encoder nos atributos

# Aplicar Label Encoding aos atributos categóricos
label_encoders = {}
x_column = banco.copy()
for column in x_column.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    x_column[column] = le.fit_transform(x_column[column])
    label_encoders[column] = le
x_column

X = x_column.drop(columns=['Genres'])
X.columns

# Escalonar os dados numéricos
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=['number']).columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
X

#y = le.fit_transform(name_column)
k = x_column['Genres'] #Definindo k para a coluna Genres
k = le.fit_transform(k)
k

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, k, test_size=0.2)

!pip install catboost

from catboost import CatBoostClassifier

# ... (Your existing code up to the model training section) ...

# Criar e treinar o modelo CatBoost
catboost_model = CatBoostClassifier(iterations=1000,  # Ajuste o número de iterações conforme necessário
                                   learning_rate=0.1,  # Ajuste a taxa de aprendizado conforme necessário
                                   depth=6,             # Ajuste a profundidade da árvore conforme necessário
                                   loss_function='MultiClass', # Define a função de perda para classificação multiclasse
                                   eval_metric='Accuracy', # Define a métrica de avaliação
                                   random_seed=42,
                                   verbose=10)

catboost_model.fit(X_train, y_train)

# Fazer previsões
y_pred_catboost_train = catboost_model.predict(X_train)
y_pred_catboost_test = catboost_model.predict(X_test)


# Métricas para o conjunto de treinamento
accuracy_catboost_train = accuracy_score(y_train, y_pred_catboost_train)
precision_catboost_train = precision_score(y_train, y_pred_catboost_train, average='weighted', zero_division=0.0)
recall_catboost_train = recall_score(y_train, y_pred_catboost_train, average='weighted', zero_division=0.0)
f1_catboost_train = f1_score(y_train, y_pred_catboost_train, average='weighted', zero_division=0.0)

# Métricas para o conjunto de teste
accuracy_catboost_test = accuracy_score(y_test, y_pred_catboost_test)
precision_catboost_test = precision_score(y_test, y_pred_catboost_test, average='weighted', zero_division=0.0)
recall_catboost_test = recall_score(y_test, y_pred_catboost_test, average='weighted', zero_division=0.0)
f1_catboost_test = f1_score(y_test, y_pred_catboost_test, average='weighted', zero_division=0.0)

# Imprimir as métricas
print(classification_report(y_test, y_pred_catboost_test, zero_division=0.0))
print("-" * 50)
print(f"Acurácia no conjunto de treinamento: {accuracy_catboost_train}")
print(f"Precisão no conjunto de treinamento: {precision_catboost_train}")
print(f"Recall no conjunto de treinamento: {recall_catboost_train}")
print(f"F1-score no conjunto de treinamento: {f1_catboost_train}")
print("-" * 50)
print(f"Acurácia no conjunto de teste: {accuracy_catboost_test}")
print(f"Precisão no conjunto de teste: {precision_catboost_test}")
print(f"Recall no conjunto de teste: {recall_catboost_test}")
print(f"F1-score no conjunto de teste: {f1_catboost_test}")