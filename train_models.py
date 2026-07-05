"""
Diabetes Early Detection — Machine Learning Model Comparison
--------------------------------------------------------------
Trains and compares three ML models (Logistic Regression, Random Forest, XGBoost)
on the Pima Indians Diabetes Dataset, following the same evaluation approach
(accuracy, precision, recall, F1, ROC-AUC) used in the published literature
this study cites (Iparraguirre-Villanueva et al., 2023; Ullah et al., 2022).

Dataset source: Pima Indians Diabetes Dataset (originally from the National
Institute of Diabetes and Digestive and Kidney Diseases), 768 female patients
of Pima Indian heritage, age 21+.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, confusion_matrix)
from xgboost import XGBClassifier
import json

RANDOM_STATE = 42

# ---------- 1. LOAD DATA ----------
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv("diabetes.csv", header=None, names=columns)

print(f"Dataset shape: {df.shape}")
print(f"Diabetic cases: {df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)")
print(f"Non-diabetic cases: {(df['Outcome']==0).sum()} ({(1-df['Outcome'].mean())*100:.1f}%)")

# ---------- 2. PREPROCESSING ----------
# Certain columns use 0 as a placeholder for missing data (biologically impossible
# to have 0 glucose, blood pressure, BMI, etc. while alive) — this is a well-known
# quirk of this specific dataset, documented in the literature that uses it.
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
missing_counts = {}
for col in zero_as_missing:
    missing_counts[col] = int((df[col] == 0).sum())
print("\nZero-value ('missing') counts per column:", missing_counts)

df_clean = df.copy()
for col in zero_as_missing:
    df_clean[col] = df_clean[col].replace(0, np.nan)
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)

# ---------- 3. TRAIN/TEST SPLIT ----------
X = df_clean.drop("Outcome", axis=1)
y = df_clean["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale features (needed for Logistic Regression; harmless for tree models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {X_train.shape[0]} patients")
print(f"Test set: {X_test.shape[0]} patients")

# ---------- 4. TRAIN MODELS ----------
models = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
}

results = {}
roc_data = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
    }

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    print(f"\n{name}:")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print(f"  ROC-AUC:   {auc:.4f}")

# ---------- 5. FEATURE IMPORTANCE (Random Forest) ----------
rf_model = models["Random Forest"]
feature_importance = dict(zip(X.columns, rf_model.feature_importances_.tolist()))
feature_importance = dict(sorted(feature_importance.items(), key=lambda x: -x[1]))
print("\nFeature importance (Random Forest):")
for feat, imp in feature_importance.items():
    print(f"  {feat}: {imp:.4f}")

# ---------- 6. SAVE RESULTS ----------
output = {
    "dataset_info": {
        "total_patients": len(df),
        "diabetic_cases": int(df['Outcome'].sum()),
        "non_diabetic_cases": int((df['Outcome']==0).sum()),
        "missing_value_counts": missing_counts,
        "train_size": len(X_train),
        "test_size": len(X_test),
    },
    "model_results": results,
    "roc_curves": roc_data,
    "feature_importance": feature_importance,
}

with open("results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nResults saved to results.json")
