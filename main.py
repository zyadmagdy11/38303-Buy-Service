"""
main.py – Run the full regression pipeline.
"""

import os
from src.data_prep import load_data, clean_data, split_data, save_preprocessing_artifacts
from src.train import train_and_select_best_model
from src.evaluate import evaluate_model

DATA_PATH = "data/ADL5ss36ORbK30dFSkAMMqiTCuCYQZep.xlsx"
MODEL_PATH = "models/regression_pipeline.pkl"
PREPROC_PATH = "models/preprocessing.pkl"
SCORING = "neg_mean_squared_error"
CV_FOLDS = 5
TEST_SIZE = 0.2

def run_pipeline():
    print("=" * 50)
    print("Regression Pipeline for Total Spent Prediction")
    print("=" * 50)
    
    print("\n[1/4] Loading and cleaning data...")
    df = load_data(DATA_PATH)
    df_clean, label_encoders, feature_columns = clean_data(df, target="Total Spent")
    
    save_preprocessing_artifacts(label_encoders, feature_columns, PREPROC_PATH)
    
    X_train, X_test, y_train, y_test = split_data(df_clean, target="Total Spent", test_size=TEST_SIZE)
    
    print("\n[2/4] Training and selecting best regression model...")
    best_model_name, best_pipeline, results_df = train_and_select_best_model(
        X_train, y_train,
        scoring=SCORING,
        cv=CV_FOLDS,
        n_jobs=-1,
        save_path=MODEL_PATH
    )
    
    print("\n[3/4] Evaluating on test set...")
    evaluate_model(X_test, y_test, model_path=MODEL_PATH, save_report=True)
    
    print("\n[4/4] Additional info:")
    if hasattr(best_pipeline.named_steps["regressor"], "feature_importances_"):
        print("Feature importances available.")
    elif hasattr(best_pipeline.named_steps["regressor"], "coef_"):
        print(f"Coefficients shape: {best_pipeline.named_steps['regressor'].coef_.shape}")
    
    print("\n✅ Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()