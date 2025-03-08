import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb


def filter_data(data, min_price, max_price, tag, category, language):
    data = data[
        (data["Price"] >= min_price) &
        (data["Price"] <= max_price) &
        (data["Tags"].str.contains(tag)) &
        (data["Categories"].str.contains(category)) &
        (data["Supported languages"].str.contains(language))
    ]

    print(data.shape)
    return data

def process_data(dataset):
    databank = dataset.copy()

    label_encoders = {}
    x_column = databank.copy()
    for column in x_column.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        x_column[column] = le.fit_transform(x_column[column])
        label_encoders[column] = le
    
    k = x_column['Genres']
    databank = x_column.drop(columns= ['Genres'])

    scaler = StandardScaler()
    numerical_cols = x_column.select_dtypes(include=['number']).columns
    databank[numerical_cols] = scaler.fit_transform(x_column[numerical_cols])

    le = LabelEncoder()
    k = le.fit_transform(k)

    return x_column, k, label_encoders


def run_xgb(X, y, label_encoders):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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

    prediction_encoded = xgb_model.predict(X_test)
    
    prediction = label_encoders['Genres'].inverse_transform(prediction_encoded)

    return prediction

def prediction(min_price, max_price, tag, category, language):
    path = os.getcwd()
    file_path = os.path.join(path, 'games_preprocessed.parquet')

    database = pd.read_parquet(file_path)

    data_filtered = filter_data(database, min_price, max_price, tag, category, language)
    X, y, label_enc = process_data(data_filtered)

    predicted = run_xgb(X, y, label_enc)

    return predicted