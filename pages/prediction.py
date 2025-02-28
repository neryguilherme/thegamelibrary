import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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

def run_rf(X, y, label_encoders):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=30, min_samples_split=10, min_samples_leaf=4, max_features='sqrt', random_state=42)
    
    rf_model.fit(X_train, y_train)

    prediction_encoded = rf_model.predict(X_test)
    
    prediction = label_encoders['Genres'].inverse_transform(prediction_encoded)

    return prediction
    

def prediction(min_price, max_price, tag, category, language):
    database = pd.read_parquet(r'..\games_preprocessed.parquet')

    data_filtered = filter_data(database, min_price, max_price, tag, category, language)
    X, y, label_enc = process_data(data_filtered)

    predicted = run_rf(X, y, label_enc)

    return predicted
