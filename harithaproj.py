# haridhaproj.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Step 1: Create synthetic dataset
# -----------------------------
np.random.seed(42)
df = pd.DataFrame({
    'status': np.random.choice(['A11','A12','A13','A14'], 1000),
    'duration': np.random.randint(4, 72, 1000),
    'credit_history': np.random.choice(['good','poor','critical'], 1000),
    'purpose': np.random.choice(['car','furniture','education','business'], 1000),
    'amount': np.random.randint(100, 20000, 1000),
    'savings': np.random.choice(['low','medium','high'], 1000),
    'employment_duration': np.random.choice(['unemployed','<1yr','1-4yr','4-7yr','>7yr'], 1000),
    'installment_rate': np.random.randint(1, 5, 1000),
    'personal_status_sex': np.random.choice(['male','female'], 1000),
    'other_debtors': np.random.choice(['none','guarantor','co-applicant'], 1000),
    'present_residence': np.random.randint(1,5,1000),
    'property': np.random.choice(['real','life','car','unknown'], 1000),
    'age': np.random.randint(18, 75, 1000),
    'other_installment_plans': np.random.choice(['none','bank','store'], 1000),
    'housing': np.random.choice(['own','rent','for free'], 1000),
    'number_credits': np.random.randint(1, 5, 1000),
    'job': np.random.randint(0, 3, 1000),
    'people_liable': np.random.randint(1, 3, 1000),
    'telephone': np.random.choice(['yes','no'], 1000),
    'foreign_worker': np.random.choice(['yes','no'], 1000),
    'credit_risk': np.random.choice(['good','bad'], 1000)
})

# -----------------------------
# Step 2: Preprocess data
# -----------------------------
target_col = 'credit_risk'
df[target_col] = df[target_col].map({'good':0, 'bad':1})  # 0 = non-default, 1 = default

# Encode categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Split data
X = df.drop(target_col, axis=1)
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 3: Train XGBoost Classifier
# -----------------------------
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# -----------------------------
# Step 4: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("\n--- Model Performance ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1-Score:", round(f1_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

# -----------------------------
# Step 5: Global Feature Importance using SHAP
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("\n--- Global Feature Importance (SHAP) ---")
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

# -----------------------------
# Step 6: Local Explanations for 5 Cases using SHAP
# -----------------------------
print("\n--- Local Explanations for 5 Case Studies ---")
sample_indices = np.random.choice(X_test.index, 5, replace=False)

for idx in sample_indices:
    print(f"\nCase {idx} - True Label:", y_test.loc[idx])
    shap.force_plot(explainer.expected_value, shap_values[idx], X_test.loc[idx], matplotlib=True)

# -----------------------------
# Step 7: Local Explanations using LIME
# -----------------------------
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['Non-Default', 'Default'],
    mode='classification'
)

for idx in sample_indices:
    print(f"\nCase {idx} - LIME explanation")
    exp = lime_explainer.explain_instance(X_test.loc[idx].values, model.predict_proba, num_features=10)
    exp.show_in_notebook(show_table=True)
    print(exp.as_list())

# -----------------------------
# Step 8: Fairness Check Example
# -----------------------------
X_test_copy = X_test.copy()
X_test_copy['pred'] = y_pred
X_test_copy['age_group'] = pd.cut(X_test_copy['age'], bins=[18,27,34,44,55,65,75])
print("\n--- Example Fairness Check by Age ---")
print(X_test_copy.groupby('age_group')['pred'].mean())
