"""
app.py – Total Spent Predictor + Sales Analytics + Training Results
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from src.data_prep import load_data, clean_data
import json
import os

# -------------------------------
# Hardcoded background colours
# -------------------------------
PLOT_BG_COLOR = "#1E1E1E"   # dark grey for plot area
PAPER_BG_COLOR = "#0D0D0D"  # near‑black for the whole figure

# -------------------------------
# Load model and preprocessing
# -------------------------------
MODEL_PATH = "models/regression_pipeline.pkl"
PREPROC_PATH = "models/preprocessing.pkl"
REPORT_PATH = "models/regression_report.json"
COMPARE_PATH = "models/model_comparison.csv"
DATA_PATH = "data/ADL5ss36ORbK30dFSkAMMqiTCuCYQZep.xlsx"

st.set_page_config(page_title="Total Spent - Complete Dashboard", layout="wide")
st.title("💰 Total Spent: Prediction, Analytics & Model Results")

@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(MODEL_PATH)
    artifacts = joblib.load(PREPROC_PATH)
    return pipeline, artifacts

@st.cache_data
def load_original_data():
    df = load_data(DATA_PATH)
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    return df

@st.cache_data
def load_training_results():
    # Load test metrics
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {"model": "Unknown", "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}
    
    # Load model comparison
    if os.path.exists(COMPARE_PATH):
        compare_df = pd.read_csv(COMPARE_PATH)
    else:
        compare_df = pd.DataFrame()
    
    return metrics, compare_df

# Load artefacts for Tab1
try:
    pipeline, artifacts = load_artifacts()
    label_encoders = artifacts["label_encoders"]
    expected_columns = artifacts["feature_columns"]
    model_available = True
except FileNotFoundError:
    model_available = False

df_original = load_original_data()
test_metrics, compare_df = load_training_results()

# Helper to apply background colours to any plotly figure
def apply_bg(fig):
    fig.update_layout(
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=PAPER_BG_COLOR,
        font=dict(color="#FFFFFF")
    )
    fig.update_xaxes(gridcolor="#333333", tickfont=dict(color="#E0E0E0"))
    fig.update_yaxes(gridcolor="#333333", tickfont=dict(color="#E0E0E0"))
    return fig

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📊 Sales Analytics", "📈 Model Training Results"])

# ============================================================
# TAB 1: SINGLE PREDICTION (same as before)
# ============================================================
with tab1:
    if not model_available:
        st.error("Model not found. Run `main.py` first to train and save the model.")
        st.stop()
    
    st.markdown("Enter transaction details and customer history to predict **Total Spent**.")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Category", label_encoders["Category"].classes_)
            payment = st.selectbox("Payment Method", label_encoders["Payment Method"].classes_)
            location = st.selectbox("Location", label_encoders["Location"].classes_)
            discount = st.selectbox("Discount Applied", [False, True])
        with col2:
            trans_date = st.date_input("Transaction Date", datetime.now())
            month = trans_date.month
            day_of_week = trans_date.weekday()
            
            st.subheader("Customer History")
            cust_avg_spent = st.number_input("Average Spend (historical)", min_value=0.0, value=50.0, step=5.0)
            cust_trans_count = st.number_input("Transaction Count (historical)", min_value=1, value=5, step=1)
            cust_recency = st.number_input("Days since last transaction", min_value=0, value=10, step=1)
        
        submitted = st.form_submit_button("Predict Total Spent")

    if submitted:
        raw_input = {
            "Category": category,
            "Payment Method": payment,
            "Location": location,
            "Discount Applied": discount,
            "Month": month,
            "DayOfWeek": day_of_week,
            "customer_avg_spent": cust_avg_spent,
            "customer_transaction_count": cust_trans_count,
            "customer_recency_days": cust_recency
        }
        input_df = pd.DataFrame([raw_input])
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))
        input_df = input_df[expected_columns]
        prediction = pipeline.predict(input_df)[0]
        st.success(f"💰 Predicted Total Spent: **${prediction:.2f}**")

# ============================================================
# TAB 2: PURE SALES ANALYTICS (unchanged)
# ============================================================
with tab2:
    st.header("Sales Analytics Dashboard")
    
    with st.expander("Filters", expanded=True):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            min_date = df_original["Transaction Date"].min().date()
            max_date = df_original["Transaction Date"].max().date()
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        with col_f2:
            all_categories = df_original["Category"].dropna().unique()
            selected_categories = st.multiselect("Category", all_categories, default=list(all_categories))
    
    date_mask = (df_original["Transaction Date"].dt.date >= start_date) & (df_original["Transaction Date"].dt.date <= end_date)
    cat_mask = df_original["Category"].isin(selected_categories) if selected_categories else True
    filtered_df = df_original[date_mask & cat_mask].copy()
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        st.stop()
    
    total_sales = filtered_df["Total Spent"].sum()
    total_transactions = filtered_df.shape[0]
    avg_spent = filtered_df["Total Spent"].mean()
    unique_categories = filtered_df["Category"].nunique()
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sales", f"${total_sales:,.0f}")
    k2.metric("Transactions", f"{total_transactions:,}")
    k3.metric("Average Spend", f"${avg_spent:,.2f}")
    k4.metric("Unique Categories", unique_categories)
    
    monthly = filtered_df.groupby(filtered_df["Transaction Date"].dt.to_period("M"))["Total Spent"].sum().reset_index()
    monthly["Month"] = monthly["Transaction Date"].astype(str)
    fig1 = px.line(monthly, x="Month", y="Total Spent", title="Monthly Sales Trend", markers=True)
    st.plotly_chart(apply_bg(fig1), use_container_width=True)
    
    cat_sales = filtered_df.groupby("Category")["Total Spent"].sum().reset_index()
    fig2 = px.bar(cat_sales, x="Category", y="Total Spent", title="Sales by Category", color="Category", text_auto='.2s')
    st.plotly_chart(apply_bg(fig2), use_container_width=True)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if "Payment Method" in filtered_df.columns:
            pay_sales = filtered_df.groupby("Payment Method")["Total Spent"].sum().reset_index()
            fig = px.bar(pay_sales, x="Payment Method", y="Total Spent", title="Sales by Payment Method", color="Payment Method", text_auto='.2s')
            st.plotly_chart(apply_bg(fig), use_container_width=True)
    with col_b:
        if "Discount Applied" in filtered_df.columns:
            discount_impact = filtered_df.groupby("Discount Applied")["Total Spent"].mean().reset_index()
            discount_impact["Discount Applied"] = discount_impact["Discount Applied"].map({0: "No Discount", 1: "Discount"})
            fig = px.bar(discount_impact, x="Discount Applied", y="Total Spent", title="Avg Spend: Discount vs No Discount", color="Discount Applied", text_auto='.2s')
            st.plotly_chart(apply_bg(fig), use_container_width=True)
    with col_c:
        if "Location" in filtered_df.columns:
            loc_sales = filtered_df.groupby("Location")["Total Spent"].sum().reset_index()
            fig = px.pie(loc_sales, values="Total Spent", names="Location", title="Sales by Location")
            st.plotly_chart(apply_bg(fig), use_container_width=True)
    
    top_cat = cat_sales.sort_values("Total Spent", ascending=False).head(10)
    st.subheader("🏆 Top 10 Categories by Sales")
    st.dataframe(top_cat, use_container_width=True)
    

# ============================================================
# TAB 3: MODEL TRAINING RESULTS
# ============================================================
with tab3:
    st.header("Model Training Results")
    
    if not model_available:
        st.error("Model not found. Run `main.py` first to train and save the model.")
        st.stop()
    
    # Display best model and test metrics
    st.subheader("✅ Best Model Performance on Test Set")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Model", test_metrics.get("model", "Unknown"))
    col_m2.metric("MSE", f"{test_metrics.get('MSE', 0):.2f}")
    col_m3.metric("RMSE", f"{test_metrics.get('RMSE', 0):.2f}")
    col_m4.metric("R²", f"{test_metrics.get('R2', 0):.3f}")
    
    # Model comparison bar chart
    if not compare_df.empty:
        st.subheader("📊 Model Comparison (Cross-Validation Neg. MSE)")
        compare_df["Model_Type"] = compare_df["model"] + " (" + compare_df["type"] + ")"
        compare_df = compare_df.sort_values("cv_score", ascending=False)
        fig_cmp = px.bar(compare_df, x="Model_Type", y="cv_score", 
                         title="CV Negative MSE (higher is better)",
                         labels={"cv_score": "Neg. MSE", "Model_Type": "Model"},
                         color="cv_score", color_continuous_scale="Viridis")
        fig_cmp.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(apply_bg(fig_cmp), use_container_width=True)
        
        with st.expander("View full comparison table"):
            st.dataframe(compare_df[["model", "type", "cv_score", "cv_std"]], use_container_width=True)
    
    # Feature coefficients with beautified names
    st.subheader("🔍 Model Coefficients (ElasticNet - Best Model)")
    
    poly_step = None
    for step_name, step in pipeline.named_steps.items():
        if step_name == "poly":
            poly_step = step
            break
    
    last_step = pipeline.named_steps["regressor"]
    if hasattr(last_step, "coef_"):
        coefs = last_step.coef_.flatten()
        
        if poly_step is not None:
            original_names = expected_columns
            raw_names = poly_step.get_feature_names_out(original_names)
            
            def beautify(name):
                name = name.replace("^2", "²")
                name = name.replace(" ", " × ")
                if len(name) > 50:
                    name = name[:47] + "..."
                return name
            
            feature_names = [beautify(n) for n in raw_names]
        else:
            feature_names = expected_columns
        
        if len(feature_names) != len(coefs):
            feature_names = [f"Feature_{i}" for i in range(len(coefs))]
        
        coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
        coef_df["Abs"] = coef_df["Coefficient"].abs()
        coef_df = coef_df.sort_values("Abs", ascending=False).head(15)
        
        fig_coef = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                          title="Top 15 Coefficients – Squared features (²) & Interactions (×)",
                          color="Coefficient", color_continuous_scale="RdBu")
        fig_coef.update_layout(height=500)
        st.plotly_chart(apply_bg(fig_coef), use_container_width=True)
        
        with st.expander("View all coefficients"):
            st.dataframe(coef_df[["Feature", "Coefficient"]], use_container_width=True)
    else:
        st.info("Coefficients not available for this model type.")
    
    # Training configuration (unchanged)
    st.subheader("⚙️ Training Configuration")
    st.markdown("""
    - **Scoring metric**: neg_mean_squared_error (lower MSE → better)
    - **Cross-validation folds**: 5
    - **Models tested**: LinearRegression, Ridge, Lasso, ElasticNet, RandomForest, XGBoost
    - **Polynomial features** (degree 2) were also tested for linear models
    - **Best model selected by highest CV negative MSE** (i.e., lowest actual MSE)
    """)
    
