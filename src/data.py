import os
import pandas as pd

def load_raw_data(file_path: str) -> pd.DataFrame:
    # Load raw CSV data from disk.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    print("Successfully loaded raw data")
    return df

def lower_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Convert all column names to lowercase and strip whitespace.
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial data cleaning tailored to the Telco Customer Churn dataset:
    - Drop customerid as it's a unique identifier, not a feature.
    - Fix totalcharges column (convert whitespace/blanks to NaN, then float).
    - Handle missing values resulting from conversion.
    """
    df = df.copy()
    
    # Drop customerid 
    df = df.drop(columns=['customerid'])
        
    # totalcharges has blank spaces for some new customers (tenure = 0)
    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
        
    # Drop rows with missing totalcharges
    initial_rows = len(df)
    df = df.dropna(subset=['totalcharges'])
    print(f"Dropped {initial_rows - len(df)} rows due to missing totalcharges.")
    
    return df

def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    # Save cleaned DataFrame to processed data directory. 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data successfully saved to: {output_path}")

if __name__ == "__main__":
    # Define relative paths based on your project structure
    RAW_DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
    PROCESSED_DATA_PATH = "data/processed/clean_churn.csv"
    
    # Execution pipeline
    raw_df = load_raw_data(RAW_DATA_PATH)
    raw_df = lower_case_columns(raw_df)
    cleaned_df = clean_data(raw_df)
    save_processed_data(cleaned_df, PROCESSED_DATA_PATH)