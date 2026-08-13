# Predictive Analytics for Credit Risk and Loan Default

Machine learning for smarter, data-driven credit risk assessment.

## Overview

This project builds a machine learning pipeline to assess credit risk and predict loan default, using historical borrower data, financial history, credit profile, and loan characteristics — to classify applicants as high-risk or low-risk.

**Pipeline:** EDA → Preprocessing → Feature Engineering → Model Building → Hyperparameter Tuning → Evaluation → Model Comparison

## Problem Statement

Traditional credit assessment is slow, manual, and struggles to process large volumes of financial data. This leads to misjudged default risk — resulting in financial losses, bad debt, ineffective loan approval decisions, and reduced operational efficiency.

## Goal & Objectives

**Goal:** Develop an AI-powered system to predict loan default risk and support faster, data-driven lending decisions.

**Objectives:**
- Collect and preprocess historical loan and financial data
- Perform EDA to identify key default risk indicators
- Develop and compare multiple ML models for risk prediction
- Target ≥85% accuracy across Accuracy, Precision, Recall, and F1-Score

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) application data, sourced from Kaggle. Contains numerical and categorical variables describing a borrower's financial background, repayment history, and loan application details — including income level, employment status, loan size, credit history, and existing debt obligations.

## Key Findings

- **Severe class imbalance:** 92% of applicants repaid their loans (TARGET = 0) vs. 8% who defaulted (TARGET = 1). Without addressing this, models default to predicting "repaid" for nearly everyone, failing to catch actual defaulters.
- **Outliers & skew:** Income, credit amount, and annuity were heavily right-skewed with outliers, requiring log transforms and robust handling.
- **Missing data:** Missingness was quantified per feature — `EXT_SOURCE_1` (56%) and `OWN_CAR_AGE` (66%) exceeded 50% missing but remained informative predictors.
- **Strongest predictors:** External credit scores (`EXT_SOURCE_3`, `EXT_SOURCE_2`, `EXT_SOURCE_1`) correlated most strongly with default, followed by applicant age, employment history, and income-related ratios.

## Feature Engineering

- **Frequency mapping** on high-cardinality variables (`ORGANIZATION_TYPE`, `OCCUPATION_TYPE`) to preserve information while reducing dimensionality
- **One-hot encoding** on nominal categorical variables (contract type, gender, income type, education, family status, housing)
- **Log transformation** on `AMT_INCOME_TOTAL` to reduce skew and soften extreme values

## Models & Results

Because the dataset was highly imbalanced, **accuracy alone was not a sufficient metric** — recall on the default class was prioritized to catch high-risk borrowers.

| Model | Accuracy | Default Recall | Default Precision | Defaults Identified |
|---|---|---|---|---|
| Logistic Regression (Tuned) | 68.5% | 68% | 16% | 3,350 / 4,949 |
| Decision Tree (Tuned) | 59.8% | 72% | 13% | 3,560 / 4,949 |
| Random Forest (Tuned) | 72.7% | 62% | 17% | 3,071 / 4,949 |
| ANN | 70.8% | 65% | — | 3,231 / 4,949 |

**Best model: Logistic Regression** — selected for its strong balance of recall (68%), interpretability, and consistency in flagging default customers. Random Forest achieved the highest raw accuracy but performed poorly on recall, making it less useful for actually catching defaulters — a good illustration of why accuracy is misleading on imbalanced data.

Class weighting was applied to reduce model bias toward the majority (non-default) class and improve detection of true defaulters.

## Conclusion

Logistic Regression and ANN provided the best balance for identifying default customers under severe class imbalance. Machine learning models like these can meaningfully support faster, more consistent credit risk assessment and lending decisions.

## Team & My Contribution

This was a group capstone project completed with:
- Stephanie Ayamga
- Kehinde Okerinde
- Omolabake Badmus
- Temiloluwa Alonge (me)

**My contribution:** EDA & data preprocessing (outlier detection, missing value analysis, distribution analysis) and model building & tuning (Logistic Regression, Decision Tree, Random Forest, hyperparameter tuning, and performance evaluation).

## Tools & Techniques

Python, scikit-learn, pandas, EDA, feature engineering, Logistic Regression, Decision Tree, Random Forest, Artificial Neural Network (ANN), hyperparameter tuning, class imbalance handling.
