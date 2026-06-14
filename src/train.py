"""
src/train.py – Enhanced regression training with RandomForest, XGBoost, and advanced tuning.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import joblib
import os

# Optional: XGBoost (install with `pip install xgboost`)
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Skipping XGBRegressor.")

RANDOM_STATE = 42

# Model configurations – keys must match the pipeline step name ("regressor")
MODEL_CONFIGS = {
    "LinearRegression": {
        "estimator": LinearRegression(),
        "params": {
            "regressor__fit_intercept": [True, False]
        }
    },
    "Ridge": {
        "estimator": Ridge(random_state=RANDOM_STATE),
        "params": {
            "regressor__alpha": [0.1, 1.0, 10.0],
            "regressor__fit_intercept": [True, False]
        }
    },
    "Lasso": {
        "estimator": Lasso(random_state=RANDOM_STATE),
        "params": {
            "regressor__alpha": [0.01, 0.1, 1.0],
            "regressor__fit_intercept": [True, False]
        }
    },
    "ElasticNet": {
        "estimator": ElasticNet(random_state=RANDOM_STATE),
        "params": {
            "regressor__alpha": [0.01, 0.1, 1.0],
            "regressor__l1_ratio": [0.5, 0.7, 0.9],
            "regressor__fit_intercept": [True]
        }
    },
    "RandomForest": {
        "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "params": {
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth": [5, 10, None],
            "regressor__min_samples_split": [2, 5],
            "regressor__min_samples_leaf": [1, 2]
        }
    }
}

# Add XGBoost if available
if XGB_AVAILABLE:
    MODEL_CONFIGS["XGBoost"] = {
        "estimator": XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=-1),
        "params": {
            "regressor__n_estimators": [100, 200],
            "regressor__learning_rate": [0.05, 0.1],
            "regressor__max_depth": [3, 5],
            "regressor__subsample": [0.8, 1.0],
            "regressor__colsample_bytree": [0.8, 1.0]
        }
    }

def build_regression_pipeline(estimator, use_poly=False, degree=2, use_scaling=True):
    """Build a pipeline with optional polynomial features and scaling."""
    steps = []
    if use_poly:
        steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    if use_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("regressor", estimator))
    return Pipeline(steps)

def train_and_select_best_model(X_train, y_train, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, save_path="models/regression_pipeline.pkl"):
    """
    Train multiple regression models with GridSearchCV and select the best.
    scoring: neg_mean_squared_error, neg_mean_absolute_error, r2
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    
    best_pipeline = None
    best_model_name = None
    best_score = -np.inf
    results = []
    
    for name, cfg in MODEL_CONFIGS.items():
        print(f"\n--- {name} ---")
        # For tree‑based models (RandomForest, XGBoost), polynomial features are rarely useful – skip.
        # For linear models, try both linear and polynomial.
        use_poly_options = [False]
        if name in ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]:
            use_poly_options = [False, True]  # also try polynomial degree 2
        
        for use_poly in use_poly_options:
            # For polynomial, we also try degree 2; could add degree 3 in a real scenario.
            pipe = build_regression_pipeline(cfg["estimator"], use_poly=use_poly, degree=2)
            param_grid = cfg["params"]
            if use_poly:
                param_grid = {**param_grid, "poly__degree": [2]}   # optionally [2,3]
            
            search = GridSearchCV(pipe, param_grid, cv=kf, scoring=scoring, n_jobs=n_jobs, verbose=0)
            search.fit(X_train, y_train)
            
            # Cross-validate the best estimator to get a stable score
            cv_scores = cross_val_score(search.best_estimator_, X_train, y_train, cv=kf, scoring=scoring, n_jobs=n_jobs)
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            print(f"  {'Polynomial' if use_poly else 'Linear'} - {scoring}: {mean_score:.4f} (±{std_score:.4f}) | best params: {search.best_params_}")
            
            results.append({
                "model": name,
                "type": "Polynomial" if use_poly else "Linear",
                "cv_score": mean_score,
                "cv_std": std_score,
                "best_params": search.best_params_
            })
            
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = f"{name}_{'Poly' if use_poly else 'Linear'}"
                best_pipeline = search.best_estimator_
    
    results_df = pd.DataFrame(results).sort_values("cv_score", ascending=False)
    print("\n--- Model Comparison (sorted by CV score) ---")
    print(results_df[["model", "type", "cv_score", "cv_std"]].to_string(index=False))
    print(f"\n✅ Best model: {best_model_name} (CV {scoring}: {best_score:.4f})")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(best_pipeline, save_path)
    print(f"Best pipeline saved to {save_path}")
    results_df.to_csv("models/model_comparison.csv", index=False)

    
    return best_model_name, best_pipeline, results_df