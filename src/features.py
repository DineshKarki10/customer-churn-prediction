import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_processed_data(file_path: str) -> pd.DataFrame:
    # Load the cleaned data generated from src/data.py.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data not found at: {file_path}. Run src/data.py first.")
    return pd.read_csv(file_path)

def preprocess_target(df: pd.DataFrame) -> tuple:
    # Separate features and target variable, mapping target to binary.
    df = df.copy()
        
    # Map churn 'Yes'/'No' to 1/0
    y = df['churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    X = df.drop(columns=['churn'])
    
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Build scikit-learn ColumnTransformer for numerical and categorical columns.
    # Identify numerical and categorical columns automatically
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    print(f"Numerical features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Create transformation pipelines for each type
    numeric_transformer = StandardScaler()
    
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

if __name__ == "__main__":
    PROCESSED_DATA_PATH = "data/processed/clean_churn.csv"
    MODEL_DIR = "models"
    
    # 1. Load data
    df = load_processed_data(PROCESSED_DATA_PATH)
    
    # 2. Split features and target
    X, y = preprocess_target(df)
    
    # 3. Train-test split (e.g., 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Build and fit preprocessor on training data only
    preprocessor = build_preprocessor(X_train)
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    print(f"Transformed training shape: {X_train_transformed.shape}")
    print(f"Transformed testing shape: {X_test_transformed.shape}")
    
    # 5. Save the preprocessor artifact so src/train.py and src/predict.py can use it
    os.makedirs(MODEL_DIR, exist_ok=True)
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor artifact successfully saved to: {preprocessor_path}")