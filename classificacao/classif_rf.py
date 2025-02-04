import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from typing import Any, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load data from Parquet file and perform initial preprocessing.
    
    Args:
        file_path: Path to the Parquet file
        target_column: Name of the column to predict
    
    Returns:
        Tuple containing features DataFrame and target Series
    """
    # Read the Parquet file
    print("Loading data from Parquet file...")
    df = pd.read_parquet(file_path)
    print(f"Dataset shape: {df.shape}")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y

def prepare_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Prepare features for model training by:
    1. Handling missing values
    2. Encoding categorical variables 
    3. Scaling numeric features using StandardScaler
    
    Args:
        X: Features DataFrame
    
    Returns:
        Tuple containing:
        - Processed DataFrame ready for model training
        - StandardScaler object (for potential inverse transformation)
    """
    # Create a copy to avoid modifying original data
    X_processed = X.copy()
    
    # Identify column types
    numeric_columns = X_processed.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = X_processed.select_dtypes(include=['object', 'category']).columns
    
    # Fill numeric missing values with median
    for col in numeric_columns:
        X_processed[col] = X_processed[col].fillna(X_processed[col].median())
    
    # Fill categorical missing values with mode
    for col in categorical_columns:
        X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0])
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        X_processed[col] = label_encoders[col].fit_transform(X_processed[col])
    
    # Initialize and apply StandardScaler to numeric features
    scaler = StandardScaler()
    X_processed[numeric_columns] = scaler.fit_transform(X_processed[numeric_columns])
    
    return X_processed, scaler

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

def train_random_forest(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestClassifier, Dict]:
    """
    Train a Random Forest classifier with limited tree size and optimized parameters.
    
    Args:
        X: Processed features DataFrame
        y: Target Series
    
    Returns:
        Tuple containing trained model and feature importance dictionary
    """
    # Initialize the model with size-constrained parameters
    rf_model = RandomForestClassifier(
        n_estimators=50,        # Number of trees
        max_leaf_nodes=15,       # Limit each tree to 20 leaves
        min_samples_split=5,     # Require more samples to split to prevent tiny leaves
        min_samples_leaf=3,      # Ensure leaves have at least 3 samples for stability
        max_features='sqrt',     # Use square root of features for each split
        bootstrap=True,          # Enable bootstrapping for better generalization
        n_jobs=-1,              # Use all available cores
        random_state=42
    )
    
    # Train the model
    print("\nTraining Random Forest model with constrained tree size (15 leaves per tree)...")
    print("Additional optimizations:")
    print("- Minimum 5 samples required for splitting")
    print("- Minimum 3 samples per leaf")
    print("- Square root feature selection at each split")
    
    rf_model.fit(X, y)
    
    # Calculate feature importance
    feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    return rf_model, feature_importance

def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate the model and display various performance metrics.
    
    Args:
        model: Trained Random Forest model
        X_test: Test features
        y_test: True test labels
        
    Returns:
        Dictionary containing all computed metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted')
    }
    
    # Print detailed metrics
    print("\nDetailed Model Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nMacro-averaged metrics (treating all classes equally):")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall: {metrics['recall_macro']:.4f}")
    print(f"F1-Score: {metrics['f1_macro']:.4f}")
    
    print("\nWeighted-averaged metrics (accounting for class imbalance):")
    print(f"Precision: {metrics['precision_weighted']:.4f}")
    print(f"Recall: {metrics['recall_weighted']:.4f}")
    print(f"F1-Score: {metrics['f1_weighted']:.4f}")
    
    # Print full classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    
    return metrics

def plot_metrics_comparison(metrics: Dict[str, float]) -> None:
    """
    Create a bar plot comparing different metrics.
    
    Args:
        metrics: Dictionary containing the computed metrics
    """
    plt.figure(figsize=(12, 6))
    
    # Select metrics to plot
    plot_metrics = {
        'Accuracy': metrics['accuracy'],
        'Precision (Weighted)': metrics['precision_weighted'],
        'Recall (Weighted)': metrics['recall_weighted'],
        'F1-Score (Weighted)': metrics['f1_weighted']
    }
    
    # Create bar plot
    plt.bar(plot_metrics.keys(), plot_metrics.values())
    plt.title('Model Performance Metrics Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on top of each bar
    for i, (metric, value) in enumerate(plot_metrics.items()):
        plt.text(i, value + 0.01, f'{value:.3f}', ha='center')
    
    plt.show()

    

def plot_tree_structure(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> None:
    """
    Plot and analyze the structure of the first tree in the forest.
    
    Args:
        model: Trained Random Forest model
        X: Feature DataFrame (for feature names)
        y: Target Series (for class names)
    """
    # Get first tree from the forest
    first_tree = model.estimators_[0]
    n_nodes = first_tree.tree_.node_count
    n_leaves = first_tree.tree_.n_leaves
    
    # Print tree statistics
    print("\nTree Structure Statistics (first tree):")
    print(f"Total nodes: {n_nodes}")
    print(f"Number of leaves: {n_leaves}")
    print(f"Average samples per leaf: {first_tree.tree_.n_node_samples[0] / n_leaves:.1f}")
    max_depth = first_tree.tree_.max_depth
    print(f"Maximum depth: {max_depth}")
    
    # Create class names if target is categorical
    if hasattr(model, 'classes_'):
        class_names = [str(c) for c in model.classes_]
    else:
        class_names = None
    
    # Create feature names list
    feature_names = list(X.columns)
    
    # Calculate figure size based on tree size
    figsize_width = min(20, max(15, n_nodes / 2))
    figsize_height = min(20, max(10, max_depth))
    
    # Plot the tree
    plt.figure(figsize=(figsize_width, figsize_height))
    plot_tree(first_tree,
             feature_names=feature_names,
             class_names=class_names,
             filled=True,
             rounded=True,
             fontsize=10,
             max_depth=5,  # Limit depth for better visualization
             proportion=True,  # Show proportions instead of counts
             precision=2)  # Decimal places for numbers in the tree
    
    plt.title("Visualization of First Tree in Random Forest", pad=20, size=14)
    plt.tight_layout()
    plt.show()
    
    # Print feature usage in first tree
    print("\nFeature Usage in First Tree:")
    feature_importance = pd.Series(
        first_tree.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
    
    print("\nTop 5 most important features in this tree:")
    for feat, imp in feature_importance.head().items():
        print(f"{feat}: {imp:.4f}")
    
def main(parquet_file: str, target_column: str) -> None:
    """
    Main function to orchestrate the entire classification process.
    
    Args:
        parquet_file: Path to the Parquet file
        target_column: Name of the target column for classification
    """
    # Load and preprocess data
    X, y = load_and_preprocess_data(parquet_file, target_column)
    
    # Prepare features
    X_processed, scaler = prepare_features(X)
    y_processed = prepare_target(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Train model and get feature importance
    model, feature_importance = train_random_forest(X_train, y_train)
        
    # Plot tree structure information
    plot_tree_structure(model, X_train, y_train)  # Updated to pass X and y
    
    # Print top 10 most important features
    print("\nTop 10 Most Important Features (Entire Forest):")
    for feature, importance in list(feature_importance.items())[:10]:
        print(f"{feature}: {importance:.4f}")

    # Evaluate model and get metrics
    print("Start Model Evaluation\n")
    metrics_train = evaluate_model(model, X_train, y_train)
    print("\nFinished Model Evaluation\n")

    # Evaluate model and get metrics
    print("Start Model Evaluation\n")
    metrics_test = evaluate_model(model, X_test, y_test)
    print("\nFinished Model Evaluation\n")

    # Plot metrics comparison
    print("Start Metrics Comparison\n")
    plot_metrics_comparison(metrics_test)
    print("\nFinished Metrics Comparison\n")

if __name__ == "__main__":
    # Replace these with your actual file path and target column name
    parquet_file_path = r"games.parquet"
    target_column_name = "Genres"
    
    main(parquet_file_path, target_column_name)