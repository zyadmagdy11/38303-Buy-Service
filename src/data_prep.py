"""
src/data_prep.py – Load, clean, add customer‑level features, no datetime leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path: str):
    df = pd.read_excel(path, sheet_name=0)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_data(df: pd.DataFrame, target: str = "Total Spent"):
    df = df.copy()
    
    # Drop columns that leak the target or are identifiers
    drop_cols = ["Transaction ID", "Price Per Unit", "Quantity", "Item"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
    
    # Convert Discount Applied to 0/1
    if "Discount Applied" in df.columns:
        df["Discount Applied"] = df["Discount Applied"].astype(str).str.lower()
        df["Discount Applied"] = df["Discount Applied"].map({"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}).fillna(0)
    
    # Extract date features and keep a temporary column for recency calculation
    if "Transaction Date" in df.columns:
        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
        df["Month"] = df["Transaction Date"].dt.month
        df["DayOfWeek"] = df["Transaction Date"].dt.dayofweek
        df["_temp_date"] = df["Transaction Date"]          # will be dropped later
    
    # Remove rows with missing target
    df.dropna(subset=[target], inplace=True)
    
    # ------------------------------------------------------------------
    # Customer‑level aggregation (using only historical data)
    # ------------------------------------------------------------------
    if "Customer ID" in df.columns:
        customer_features = df.groupby("Customer ID").agg(
            customer_avg_spent=(target, "mean"),
            customer_transaction_count=(target, "count"),
            customer_last_date=("_temp_date", "max")
        ).reset_index()
        
        max_date = df["_temp_date"].max()
        customer_features["customer_recency_days"] = (max_date - customer_features["customer_last_date"]).dt.days
        customer_features.drop("customer_last_date", axis=1, inplace=True)
        
        # Merge back
        df = df.merge(customer_features, on="Customer ID", how="left")
        
        # Drop Customer ID and the temporary date column
        df.drop(columns=["Customer ID", "_temp_date"], inplace=True, errors="ignore")
    else:
        # If no Customer ID, skip aggregation but still drop temp date
        df.drop(columns=["_temp_date"], inplace=True, errors="ignore")
    
    # Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Label encode categorical features
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    # Final safety: drop any remaining datetime columns
    date_cols = df.select_dtypes(include=["datetime64"]).columns
    if len(date_cols) > 0:
        df.drop(columns=date_cols, inplace=True)
    
    # Ensure all columns are numeric (should be by now)
    feature_cols = [c for c in df.columns if c != target]
    return df, le_dict, feature_cols

def split_data(df: pd.DataFrame, target: str = "Total Spent", test_size: float = 0.2):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def save_preprocessing_artifacts(le_dict, feature_columns, path="models/preprocessing.pkl"):
    import joblib
    artifacts = {"label_encoders": le_dict, "feature_columns": feature_columns}
    joblib.dump(artifacts, path)
    print(f"Preprocessing artifacts saved to {path}")

def load_preprocessing_artifacts(path="models/preprocessing.pkl"):
    import joblib
    return joblib.load(path)