Interpretable Credit Risk Scoring with SHAP & LIME
Project Overview

This project demonstrates the development and interpretation of a machine learning model for predicting loan default risk. Using a synthetic dataset (or the UCI German Credit dataset), an XGBoost classifier is trained to predict whether a borrower is likely to default.

The project focuses not only on predictive accuracy but also on model interpretability and fairness, making it suitable for applications in regulated industries like banking.

Features

Data preprocessing for categorical and numerical features.

Training an XGBoost classifier for credit risk prediction.

Model evaluation with Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

Global feature importance visualization using SHAP.

Local explanations of individual predictions using SHAP and LIME.

Fairness analysis to identify potential prediction disparities across demographic groups.

Tech Stack

Python 3.13

Pandas for data manipulation

NumPy for numerical operations

Scikit-learn for preprocessing and metrics

XGBoost for model training

SHAP for explainable AI (global & local interpretations)

LIME for local explanations

Matplotlib for visualizations

Installation

Clone the repository or download the project files.

Navigate to the project folder in your terminal.

Install dependencies:

pip install -r requirements.txt

Usage

Save the script as haridhaproj.py.

Run the script:

py haridhaproj.py


The script will:

Generate a synthetic dataset.

Train an XGBoost model.

Display model performance metrics.

Show global feature importance using SHAP.

Provide local explanations for 5 selected cases using SHAP and LIME.

Perform a simple fairness check based on age groups.

Sample Output
--- Model Performance ---
Accuracy: 0.75
Precision: 0.80
Recall: 0.8571
F1-Score: 0.8276
ROC-AUC: 0.7604

--- Global Feature Importance (SHAP) ---
[Bar plot of feature importance]

--- Local Explanations for 5 Cases ---
Case 1 - True Label: 1
SHAP values: [Force plot]
LIME explanation: [List of top contributing features]
...


Fairness Check Example by Age

age_group
(18.999, 27.0]    0.654545
(27.0, 34.0]      0.755102
(34.0, 44.0]      0.836735
...

Project Benefits

Helps risk committees understand why the model predicts defaults or non-defaults.

Provides transparent AI insights for compliance in the finance industry.

Demonstrates interpretable machine learning using SHAP and LIME.

License

This project is released under the MIT License.
