#  Telco Customer Churn Prediction & Deployment Pipeline

An end-to-end, production-ready machine learning pipeline and interactive web application designed to predict customer churn for telecommunications companies. This project emphasizes clean code structure, feature preprocessing pipelines, model interpretability, and containerized deployment using Docker and Streamlit.

---

##  Problem Statement
Customer churn is a critical challenge for telecommunications companies, leading to direct revenue loss and increased acquisition costs. The objective of this project is to build a predictive machine learning model and interactive web application that identifies subscribers at high risk of churning, enabling proactive customer retention strategies.

---

##  Data Source
* **Dataset:** IBM Telco Customer Churn Dataset (publicly available via Kaggle).
* **Features:** Includes customer demographic details (gender, senior citizen status, partners, dependents), account information (tenure, contract type, paperless billing, payment method, monthly and total charges), and subscribed phone/internet services.

---

##  Model Performance & Results
After benchmarking multiple algorithms (including Random Forest and Gradient Boosting), **Logistic Regression** (configured with balanced class weights) was selected as the final production model:
* **Recall (Churn Class):** **0.80** — Successfully captures 80% of actual customers who churn, minimizing missed churners[cite: 1].
* **Precision (Churn Class):** **0.49**.
* **ROC-AUC Score:** **~0.835** — Demonstrates strong overall ranking capability and generalizability.

---

##  Project Directory Structure

```text
customer-churn-prediction/
│
├── app/
│   └── app.py                # Streamlit user interface
├── data/
│   ├── processed/            # Cleaned and processed dataset artifacts
│   └── raw/                  # Original Telco Customer Churn CSV dataset
├── models/
│   ├── model.joblib          # Trained Logistic Regression model artifact
│   └── preprocessor.joblib   # Fitted Scikit-Learn ColumnTransformer
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   └── 02_modeling.ipynb     # Model training, benchmarking, and selection
├── src/
│   ├── data.py               # Data cleaning and ingestion scripts
│   ├── features.py           # Feature engineering and preprocessing pipeline
│   ├── train.py              # Model training script
│   └── predict.py            # Inference wrapper class
│
├── Dockerfile                # Docker container configuration
├── docker-compose.yml        # Multi-container orchestration setup
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
```

## How to Run the Project

### Option 1: Running with Docker (Recommended)
Make sure you have **Docker Desktop** installed and running on your machine, then execute the following command in your project root directory:

```bash
docker-compose up --build
```

### Option 2: Running Locally (Manual Setup)
1. Set up a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
- On Windows 
.venv\Scripts\Activate.ps1

- On Mac/Linux
source .venv/bin/activate

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model (to generate model artifacts):
```bash
python src/train.py
```

5. Launch the Streamlit app locally:
```bash
streamlit run app/app.py
```
Author
Dinesh Karki