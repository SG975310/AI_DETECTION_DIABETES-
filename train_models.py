"""
Diabetes Early Detection — Machine Learning Model Comparison
==============================================================

Trains and compares three ML models (Logistic Regression, Random Forest,
XGBoost) on the Pima Indians Diabetes Dataset, following the same
evaluation approach (accuracy, precision, recall, F1, ROC-AUC) used in
the published literature this project builds on (Iparraguirre-Villanueva
et al., 2023; Ullah et al., 2022).

Dataset: Pima Indians Diabetes Dataset (National Institute of Diabetes
and Digestive and Kidney Diseases), 768 female patients of Pima Indian
heritage, age 21+.

Usage:
    python src/train_models.py
"""

import json
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score,
                              roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                      train_test_split)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DATA_PATH = "data/diabetes.csv"
RESULTS_PATH = "results/results.json"

COLUMN_NAMES: List[str] = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]

# Columns where a value of 0 is a data-entry placeholder for "missing",
# not a real physiological measurement (0 blood pressure isn't possible
# in a living patient). This is a documented quirk of this dataset.
ZERO_AS_MISSING_COLUMNS: List[str] = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
]


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and attach proper column names."""
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    return df


def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Log and return basic dataset statistics (size, class balance, missingness)."""
    n_total = len(df)
    n_diabetic = int(df["Outcome"].sum())
    n_healthy = n_total - n_diabetic

    logger.info("Dataset shape: %s", df.shape)
    logger.info("Diabetic cases: %d (%.1f%%)", n_diabetic, n_diabetic / n_total * 100)
    logger.info("Non-diabetic cases: %d (%.1f%%)", n_healthy, n_healthy / n_total * 100)

    missing_counts = {col: int((df[col] == 0).sum()) for col in ZERO_AS_MISSING_COLUMNS}
    logger.info("Zero-value ('missing') counts per column: %s", missing_counts)

    return {
        "total_patients": n_total,
        "diabetic_cases": n_diabetic,
        "non_diabetic_cases": n_healthy,
        "missing_value_counts": missing_counts,
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace placeholder zeros with NaN, then impute using the column
    median. The median is used (rather than the mean) because several
    of these columns are right-skewed, and the median is more robust
    to outliers in that situation.
    """
    df_clean = df.copy()
    for col in ZERO_AS_MISSING_COLUMNS:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        median_value = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_value)
    return df_clean


def split_and_scale(
    df_clean: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """
    Split into train/test sets (stratified to preserve class balance)
    and produce a scaled version of the features for models that need
    standardized inputs (Logistic Regression). Tree-based models use
    the unscaled features directly, since scaling doesn't affect their
    splits.
    """
    X = df_clean.drop("Outcome", axis=1)
    y = df_clean["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


def build_models() -> Dict[str, ClassifierMixin]:
    """Return a dict of the three models being compared, with a fixed
    random seed so results are reproducible run-to-run."""
    return {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss"
        ),
    }


def evaluate_model(
    model: ClassifierMixin,
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """
    Fit one model and compute its full evaluation metrics on the held-out
    test set: accuracy, precision, recall, F1, ROC-AUC, and the
    confusion matrix. Logistic Regression uses the standardized
    features; the tree-based models use the raw features.
    """
    uses_scaled_input = (name == "Logistic Regression")
    train_X = X_train_scaled if uses_scaled_input else X_train
    test_X = X_test_scaled if uses_scaled_input else X_test

    model.fit(train_X, y_train)
    y_pred = model.predict(test_X)
    y_proba = model.predict_proba(test_X)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall": round(recall_score(y_test, y_pred) * 100, 2),
        "f1_score": round(f1_score(y_test, y_pred) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    return metrics, roc_data


def run_cross_validation(
    models: Dict[str, ClassifierMixin], X: pd.DataFrame, y: pd.Series, n_folds: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    Run stratified k-fold cross-validation for each model as a
    robustness check alongside the single train/test split above.
    This reports how stable each model's accuracy is across different
    data splits, rather than relying on one split alone.
    """
    cv_results: Dict[str, Dict[str, Any]] = {}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        if name == "Logistic Regression":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="accuracy")
        else:
            scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

        cv_results[name] = {
            "mean_accuracy": round(scores.mean() * 100, 2),
            "std_accuracy": round(scores.std() * 100, 2),
            "fold_scores": [round(s * 100, 2) for s in scores],
        }
        logger.info(
            "%s — %d-fold CV accuracy: %.2f%% (+/- %.2f%%)",
            name, n_folds, scores.mean() * 100, scores.std() * 100,
        )

    return cv_results


def get_feature_importance(
    rf_model: RandomForestClassifier, feature_names: List[str]
) -> Dict[str, float]:
    """Extract and sort Random Forest feature importances."""
    importance = dict(zip(feature_names, rf_model.feature_importances_.tolist()))
    return dict(sorted(importance.items(), key=lambda item: -item[1]))


def main() -> None:
    df = load_dataset()
    dataset_info = summarize_dataset(df)

    df_clean = clean_dataset(df)
    (X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled) = split_and_scale(df_clean)

    dataset_info["train_size"] = len(X_train)
    dataset_info["test_size"] = len(X_test)
    logger.info("Training set: %d patients", len(X_train))
    logger.info("Test set: %d patients", len(X_test))

    models = build_models()
    results: Dict[str, Any] = {}
    roc_curves: Dict[str, Any] = {}

    logger.info("--- Train/Test Split Evaluation ---")
    for name, model in models.items():
        metrics, roc_data = evaluate_model(
            model, name, X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled
        )
        results[name] = metrics
        roc_curves[name] = roc_data
        logger.info("%s: %s", name, {
            k: v for k, v in metrics.items() if k != "confusion_matrix"
        })

    logger.info("--- 5-Fold Cross-Validation (robustness check) ---")
    X_full = df_clean.drop("Outcome", axis=1)
    y_full = df_clean["Outcome"]
    fresh_models = build_models()  # untrained copies, so CV doesn't reuse fitted state
    cv_results = run_cross_validation(fresh_models, X_full, y_full)

    feature_importance = get_feature_importance(
        models["Random Forest"], X_train.columns.tolist()
    )
    logger.info("Feature importance (Random Forest): %s", feature_importance)

    output = {
        "dataset_info": dataset_info,
        "model_results": results,
        "cross_validation": cv_results,
        "roc_curves": roc_curves,
        "feature_importance": feature_importance,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
