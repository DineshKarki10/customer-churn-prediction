import os
import joblib
import pandas as pd

class ChurnPredictor:
    def __init__(self, model_dir: str = "models"):
        # Load artifacts once when the predictor is instantiated
        self.model = joblib.load(os.path.join(model_dir, "model.joblib"))
        self.preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))

    def predict(self, input_data: dict) -> dict:
        # Takes a dictionary of customer features and returns prediction & probability."""
        df = pd.DataFrame([input_data])
        df.columns = df.columns.str.lower().str.strip()
        
        X_processed = self.preprocessor.transform(df)
        prediction = self.model.predict(X_processed)[0]
        probability = self.model.predict_proba(X_processed)[0][1]
        
        return {
            "prediction": "Churn" if prediction == 1 else "Stay",
            "churn_probability": float(round(probability, 4))
        }

if __name__ == "__main__":
    # Quick sanity check test when running "python src/predict.py" directly
    sample_customer = {
        "gender": "Female",
        "seniorcitizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": 1,
        "phoneservice": "No",
        "multiplelines": "No phone service",
        "internetservice": "DSL",
        "onlinesecurity": "No",
        "onlinebackup": "Yes",
        "deviceprotection": "No",
        "techsupport": "No",
        "streamingtv": "No",
        "streamingmovies": "No",
        "contract": "Month-to-month",
        "paperlessbilling": "Yes",
        "paymentmethod": "Electronic check",
        "monthlycharges": 29.85,
        "totalcharges": 29.85
    }
    
    predictor = ChurnPredictor()
    result = predictor.predict(sample_customer)
    print("Test Prediction Result:", result)