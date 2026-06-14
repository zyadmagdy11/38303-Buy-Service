"""
src/evaluate.py – Evaluate regression model, compute metrics, save report.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No pipeline found at '{path}'. Run main.py first.")
    return joblib.load(path)

def evaluate_model(X_test, y_test, model_path, save_report=True, report_path="models/regression_report.json"):
    pipeline = load_model(model_path)
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    last_step_name = list(pipeline.named_steps.keys())[-1]
    model_name = type(pipeline.named_steps[last_step_name]).__name__
    
    metrics = {
        "model": model_name,
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
    }
    
    print("\n--- Test Metrics ---")
    for k, v in metrics.items():
        print(f"{k:8}: {v}")
    
    if save_report:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Report saved to {report_path}")
    
    return metrics

def predict_new(pipeline, X_new):
    return pipeline.predict(X_new)