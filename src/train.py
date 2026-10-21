import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

def train_model():
    # Define paths
    PROCESSED_DATA_PATH = "data/processed/clean_churn.csv"
    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
    PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")

    # Load processed data
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}. Run src/data.py first.")
    
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Separate features and target
    y = df['churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    X = df.drop(columns=['churn'])

    # Train-test split (matching the seed and stratify from the pipeline)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load the preprocessor artifact
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}. Run src/features.py first.")
    
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Transform data
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Initialize and train the chosen model (Logistic Regression)
    print("Training Logistic Regression model")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train_transformed, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test_transformed)
    y_prob = model.predict_proba(X_test_transformed)[:, 1]
    
    test_auc = roc_auc_score(y_test, y_prob)
    print(f"Model successfully trained! Test ROC-AUC: {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model artifact
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model artifact successfully saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()