"""
Loan Default Risk — Demo App
-----------------------------
Run with:  streamlit run streamlit_app.py

Requires these files in the same folder (created by the notebook's "Load Model" cells):
  - loan_default_model.pkl
  - loan_default_scaler.pkl
  - loan_default_columns.pkl
  - loan_default_reference_row.pkl
"""

import streamlit as st
import joblib
import pandas as pd

NAVY, MAROON, SLATE = "#1E2761", "#B23A48", "#5A6478"

st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="\U0001F4CA", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("loan_default_model.pkl")
    scaler = joblib.load("loan_default_scaler.pkl")
    columns = joblib.load("loan_default_columns.pkl")
    reference_row = joblib.load("loan_default_reference_row.pkl")
    return model, scaler, columns, reference_row


def predict_default(user_inputs: dict, model, scaler, columns, reference_row) -> dict:
    row = reference_row.copy()
    for feature, value in user_inputs.items():
        if feature in row.index:
            row[feature] = value
    row = row[columns]
    row_scaled = scaler.transform([row.values])
    prob_default = model.predict_proba(row_scaled)[0][1]
    if prob_default >= 0.5:
        category = "High Risk"
    elif prob_default >= 0.25:
        category = "Medium Risk"
    else:
        category = "Low Risk"
    return {"risk_score": float(prob_default), "category": category}


st.markdown(f"<h1 style='color:{NAVY};'>Loan Default Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{SLATE};'>Enter an applicant's details below. Fields left at default use a "
    f"typical applicant's value from the training data.</p>",
    unsafe_allow_html=True,
)

try:
    model, scaler, columns, reference_row = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run the 'Load Model' cells in the notebook first to generate:\n"
        "loan_default_model.pkl, loan_default_scaler.pkl, loan_default_columns.pkl, "
        "loan_default_reference_row.pkl — then place them in this same folder."
    )
    st.stop()

st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    ext_source_1 = st.slider("EXT_SOURCE_1 (external credit score)", 0.0, 1.0, float(reference_row.get("EXT_SOURCE_1", 0.5)), 0.01)
    ext_source_2 = st.slider("EXT_SOURCE_2 (external credit score)", 0.0, 1.0, float(reference_row.get("EXT_SOURCE_2", 0.5)), 0.01)
    ext_source_3 = st.slider("EXT_SOURCE_3 (external credit score)", 0.0, 1.0, float(reference_row.get("EXT_SOURCE_3", 0.5)), 0.01)
    days_birth = st.number_input("Age (years)", 18, 90, int(reference_row.get("DAYS_BIRTH", 40)))
    days_employed = st.number_input("Years employed", 0, 50, int(reference_row.get("DAYS_EMPLOYED", 5)))
    amt_credit = st.number_input("Loan / credit amount ($)", 0, 5_000_000, int(reference_row.get("AMT_CREDIT", 500000)), step=10000)

with col2:
    amt_annuity = st.number_input("Annual loan payment / annuity ($)", 0, 500_000, int(reference_row.get("AMT_ANNUITY", 25000)), step=1000)
    credit_income_ratio = st.number_input("Credit-to-income ratio", 0.0, 20.0, float(reference_row.get("CREDIT_INCOME_RATIO", 3.0)), 0.1)
    annuity_income_ratio = st.number_input("Annuity-to-income ratio", 0.0, 5.0, float(reference_row.get("ANNUITY_INCOME_RATIO", 0.2)), 0.01)
    days_id_publish = st.number_input("Years since ID document issued", 0, 30, int(reference_row.get("DAYS_ID_PUBLISH", 5)))
    days_registration = st.number_input("Years since registration change", 0, 30, int(reference_row.get("DAYS_REGISTRATION", 5)))
    days_last_phone_change = st.number_input("Days since last phone change", 0, 5000, int(reference_row.get("DAYS_LAST_PHONE_CHANGE", 500)))

if st.button("Predict Risk", type="primary", use_container_width=True):
    user_inputs = {
        "EXT_SOURCE_1": ext_source_1,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "CREDIT_INCOME_RATIO": credit_income_ratio,
        "ANNUITY_INCOME_RATIO": annuity_income_ratio,
        "DAYS_ID_PUBLISH": days_id_publish,
        "DAYS_REGISTRATION": days_registration,
        "DAYS_LAST_PHONE_CHANGE": days_last_phone_change,
    }
    result = predict_default(user_inputs, model, scaler, columns, reference_row)

    color = MAROON if result["category"] == "High Risk" else ("#B8860B" if result["category"] == "Medium Risk" else NAVY)
    st.markdown("---")
    st.markdown(f"<h2 style='color:{color};'>{result['category']}</h2>", unsafe_allow_html=True)
    st.metric("Predicted default probability", f"{result['risk_score']*100:.1f}%")
    st.progress(min(result["risk_score"], 1.0))

st.markdown("---")
st.caption(
    "Model: Logistic Regression (class-balanced). Only the model's top predictors are exposed here; "
    "all other applicant fields are held at their training-set median."
)
