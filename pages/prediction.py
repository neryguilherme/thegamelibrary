import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle


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

def process_genres(genres_str):
    if not isinstance(genres_str, str):
        return "Outros"
    
    genres = set(g.strip() for g in genres_str.split(','))
    valid_genres = {"Indie", "Action", "Casual"}
    
    filtered_genres = sorted(valid_genres.intersection(genres))
    if len(filtered_genres) < len(genres):
        filtered_genres.append("Outros")
    
    return ", ".join(filtered_genres)

def process_data(dataset):
    x_column = dataset.copy()
    
    x_column['Genres'] = x_column['Genres'].apply(process_genres)
    label_encoders2 = {}

    for filename in os.listdir('label_encoders'):
        if filename.endswith('.pkl'):
            key = filename[:-4]  # Remove '.pkl' para obter o nome da chave
            filepath = os.path.join('label_encoders', filename)
            with open(filepath, 'rb') as f:
                label_encoders2[key] = pickle.load(f)
    
    for column_name, encoder3 in label_encoders2.items():
        if column_name in x_column.columns:
            x_column[column_name] = encoder3.transform(x_column[column_name])
    
    k = x_column['Genres']
    x_column = x_column.drop(columns= ['Genres'])

    scaler = StandardScaler()
    numerical_cols = x_column.select_dtypes(include=['number']).columns
    x_column[numerical_cols] = scaler.fit_transform(x_column[numerical_cols])

    le = LabelEncoder()
    k = le.fit_transform(k)

    return x_column, k, label_encoders2


def run_xgb(X, y, label_encoders):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    path = os.getcwd()
    xgb_model_path = os.path.join(path, 'xgb_model.pkl')
    xgb_model = pickle.load(open(xgb_model_path, 'rb'))
    
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
