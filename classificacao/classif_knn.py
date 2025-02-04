import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Generator
import warnings
from sklearn.decomposition import PCA
import psutil
import gc
from tqdm import tqdm

warnings.filterwarnings('ignore')

def get_memory_usage() -> str:
    """
    Get current memory usage of the Python process.
    
    Returns:
        String describing current memory usage
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return f"{memory_info.rss / 1024 / 1024:.2f} MB"

def batch_generator(df: pd.DataFrame, batch_size: int) -> Generator:
    """
    Generate batches of data to process large datasets efficiently.
    
    Args:
        df: Input DataFrame
        batch_size: Size of each batch
    
    Yields:
        Batch of data as DataFrame
    """
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        yield df.iloc[start_idx:end_idx]

def load_and_preprocess_data(file_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load data from Parquet file with memory efficiency considerations.
    
    Args:
        file_path: Path to the Parquet file
        target_column: Name of the column to predict
    
    Returns:
        Tuple containing features DataFrame and target Series
    """
    print(f"Initial memory usage: {get_memory_usage()}")
    print("Loading data from Parquet file...")
    
    # Read the entire Parquet file
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        raise
    
    print(f"Dataset shape: {df.shape}")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Clear memory
    del df
    gc.collect()
    print(f"Memory usage after loading: {get_memory_usage()}")
    
    return X, y

def prepare_features(X: pd.DataFrame, batch_size: int = 5000) -> Tuple[pd.DataFrame, StandardScaler, Dict[str, LabelEncoder]]:
    """
    Prepare features in batches to manage memory usage.
    
    Args:
        X: Features DataFrame
        batch_size: Size of batches for processing
    
    Returns:
        Tuple containing processed DataFrame, scaler object, and dictionary of encoders
    """
    print("\nPreparing features in batches...")
    print(f"Initial memory usage: {get_memory_usage()}")
    
    # Initialize containers
    X_processed = pd.DataFrame()
    label_encoders = {}
    
    # Handle categorical and numeric columns
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    
    # Process in batches
    for batch in tqdm(batch_generator(X, batch_size), total=len(X)//batch_size + 1):
        # Handle missing values
        for col in numeric_columns:
            batch[col] = batch[col].fillna(X[col].median())
        
        for col in categorical_columns:
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder()
                label_encoders[col].fit(X[col].dropna())
            batch[col] = batch[col].fillna(X[col].mode()[0])
            batch[col] = label_encoders[col].transform(batch[col])
        
        X_processed = pd.concat([X_processed, batch], ignore_index=True)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_processed = pd.DataFrame(
        scaler.fit_transform(X_processed),
        columns=X_processed.columns
    )
    
    # Clear memory
    gc.collect()
    print(f"Memory usage after feature preparation: {get_memory_usage()}")
    
    return X_processed, scaler, label_encoders

def prepare_target(y: pd.Series) -> pd.Series:
    """
    Prepare target variable for model training by encoding categorical values.

    Args:
        y: Target Series

    Returns:
        Processed target Series ready for model training
    """
    # Create a copy to avoid modifying original data
    y_processed = y.copy()

    # Encode categorical values if the target is categorical
    if pd.api.types.is_object_dtype(y_processed) or pd.api.types.is_categorical_dtype(y_processed):
        label_encoder = LabelEncoder()
        y_processed = label_encoder.fit_transform(y_processed)

    return y_processed

def apply_pca_in_batches(X: pd.DataFrame, variance_threshold: float = 0.95, 
                        batch_size: int = 5000) -> Tuple[pd.DataFrame, PCA]:
    """
    Apply PCA using batch processing to manage memory.
    
    Args:
        X: Processed features DataFrame
        variance_threshold: Minimum explained variance to retain
        batch_size: Size of batches for processing
    
    Returns:
        Tuple containing transformed features and PCA object
    """
    print("\nApplying PCA in batches...")
    print(f"Initial memory usage: {get_memory_usage()}")
    
    # Initialize PCA
    pca = PCA(n_components=variance_threshold, svd_solver='full')
    
    # Fit PCA on first batch to get components
    first_batch = next(batch_generator(X, batch_size))
    pca.fit(first_batch)
    
    # Transform data in batches
    transformed_chunks = []
    for batch in tqdm(batch_generator(X, batch_size), total=len(X)//batch_size + 1):
        transformed_chunk = pca.transform(batch)
        transformed_chunks.append(transformed_chunk)
    
    # Combine transformed chunks
    X_pca = np.vstack(transformed_chunks)
    X_pca = pd.DataFrame(
        X_pca,
        columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
    )
    
    # Clear memory
    del transformed_chunks
    gc.collect()
    
    print(f"Reduced dimensions from {X.shape[1]} to {X_pca.shape[1]} features")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Memory usage after PCA: {get_memory_usage()}")
    
    return X_pca, pca

def find_optimal_k(X_train: pd.DataFrame, X_val: pd.DataFrame, 
                  y_train: pd.Series, y_val: pd.Series,
                  batch_size: int = 5000) -> int:
    """
    Find optimal K value using batch processing for cross-validation.
    
    Args:
        X_train, X_val: Training and validation features
        y_train, y_val: Training and validation targets
        batch_size: Size of batches for processing
    
    Returns:
        Optimal K value
    """
    print("\nFinding optimal K value using batch processing...")
    print(f"Initial memory usage: {get_memory_usage()}")
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19]
    }
    
    # Initialize KNN with batch processing
    knn = KNeighborsClassifier(weights='distance', n_jobs=-1)
    
    # Custom batch processing for cross-validation
    scores = []
    for k in tqdm(param_grid['n_neighbors']):
        knn.set_params(n_neighbors=k)
        batch_scores = []
        
        for batch_idx in range(0, len(X_train), batch_size):
            batch_end = min(batch_idx + batch_size, len(X_train))
            X_batch = X_train.iloc[batch_idx:batch_end]
            y_batch = y_train.iloc[batch_idx:batch_end]
            
            knn.fit(X_batch, y_batch)
            score = knn.score(X_val, y_val)
            batch_scores.append(score)
        
        scores.append(np.mean(batch_scores))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(param_grid['n_neighbors'], scores)
    plt.xlabel('K Value')
    plt.ylabel('Validation Accuracy')
    plt.title('K Value vs Model Performance')
    plt.show()
    
    optimal_k = param_grid['n_neighbors'][np.argmax(scores)]
    print(f"Optimal K value: {optimal_k}")
    print(f"Memory usage after K optimization: {get_memory_usage()}")
    
    return optimal_k

def train_knn_in_batches(X_train: pd.DataFrame, y_train: pd.Series, 
                        k: int, batch_size: int = 5000) -> KNeighborsClassifier:
    """
    Train KNN classifier using batch processing.
    
    Args:
        X_train: Training features
        y_train: Training target
        k: Optimal K value
        batch_size: Size of batches for processing
    
    Returns:
        Trained KNN model
    """
    print("\nTraining KNN classifier in batches...")
    print(f"Initial memory usage: {get_memory_usage()}")
    
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',
        metric='euclidean',
        n_jobs=-1
    )
    
    # Train on full dataset but monitor memory
    knn.fit(X_train, y_train)
    
    print(f"Memory usage after training: {get_memory_usage()}")
    return knn

def evaluate_model(model: KNeighborsClassifier, X_test: pd.DataFrame, 
                  y_test: pd.Series, batch_size: int = 5000) -> None:
    """
    Evaluate the KNN model's performance with specific metrics (Accuracy, Precision,
    F1-score, and Recall) using batch processing to manage memory efficiently.
    
    Args:
        model: Trained KNN model
        X_test: Test features
        y_test: True test labels
        batch_size: Size of batches for processing predictions
    """
    print("\nEvaluating model in batches...")
    print(f"Initial memory usage: {get_memory_usage()}")
    
    # Initialize lists to store predictions
    all_predictions = []
    
    # Make predictions in batches to manage memory
    print("Making predictions in batches...")
    for batch in tqdm(batch_generator(X_test, batch_size), 
                     total=len(X_test)//batch_size + 1):
        # Get predictions for current batch
        batch_predictions = model.predict(batch)
        all_predictions.extend(batch_predictions)
        
        # Clear memory after each batch
        gc.collect()
    
    # Convert predictions list to numpy array
    y_pred = np.array(all_predictions)
    
    
    
    # Calculate overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Handle multi-class cases with weighted averaging
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print metrics in a formatted way
    print("\nModel Performance Metrics:")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("-" * 40)
    
    # Calculate and display per-class metrics
    classes = sorted(list(set(y_test)))
    print("\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 60)
    
    for cls in classes:
        # Calculate binary metrics for each class
        cls_precision = precision_score(y_test == cls, y_pred == cls)
        cls_recall = recall_score(y_test == cls, y_pred == cls)
        cls_f1 = f1_score(y_test == cls, y_pred == cls)
        
        print(f"{str(cls):<15} {cls_precision:>10.4f} {cls_recall:>10.4f} {cls_f1:>10.4f}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    print(f"\nFinal memory usage: {get_memory_usage()}")

def main(parquet_file: str, target_column: str) -> None:
    """
    Main function with memory optimization for large datasets.
    
    Args:
        parquet_file: Path to the Parquet file
        target_column: Name of the target column for classification
    """
    # Set batch size based on available memory
    total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
    batch_size = int(min(5000, total_memory * 1000))  # Adjust batch size based on RAM
    
    print(f"Total system memory: {total_memory:.2f} GB")
    print(f"Using batch size: {batch_size}")
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(parquet_file, target_column)
    
    # Prepare features
    X_processed, scaler, encoders = prepare_features(X, batch_size=batch_size)
    y_processed = prepare_target(y)
    # Clear memory
    del X
    gc.collect()
    
    # Apply PCA
    X_pca, pca = apply_pca_in_batches(X_processed, batch_size=batch_size)
    
    # Clear memory
    del X_processed
    gc.collect()
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_pca, y_processed, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Find optimal K value
    optimal_k = find_optimal_k(X_train, X_val, y_train, y_val, batch_size=batch_size)
    
    # Train model
    model = train_knn_in_batches(X_train, y_train, optimal_k, batch_size=batch_size)
    
    #Evaluate Training Set
    evaluate_model(model, X_train,y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    parquet_file_path = "games.parquet"
    target_column_name = "Positive"
    
    
    main(parquet_file_path, target_column_name)