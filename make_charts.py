"""
Generates all visualizations for the diabetes ML project: model
comparison, ROC curves, feature importance, correlation heatmap, class
distribution, and confusion matrices.

Usage:
    python src/make_charts.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_PATH = "results/results.json"
DATA_PATH = "data/diabetes.csv"
IMAGES_DIR = "images"

BLUE, GREEN, AMBER, RED = "#2a78d6", "#199e70", "#eda100", "#e34948"
plt.rcParams["font.family"] = "DejaVu Sans"


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def plot_model_comparison(results):
    """Bar chart comparing accuracy/precision/recall/F1 across models."""
    models = list(results["model_results"].keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors = [BLUE, GREEN, AMBER, RED]

    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = [results["model_results"][m][metric] for m in models]
        ax.bar(x + i * width, values, width, label=label, color=color)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Model Performance Comparison — Diabetes Prediction",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/chart_model_comparison.png", dpi=150)
    plt.close()


def plot_roc_curves(results):
    """ROC curve for each model, with AUC in the legend."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"Logistic Regression": BLUE, "Random Forest": GREEN, "XGBoost": RED}

    for model, roc in results["roc_curves"].items():
        auc = results["model_results"][model]["roc_auc"]
        ax.plot(roc["fpr"], roc["tpr"], label=f"{model} (AUC = {auc:.3f})",
                 color=colors[model], linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("ROC Curves — Diabetes Prediction Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/chart_roc_curves.png", dpi=150)
    plt.close()


def plot_feature_importance(results):
    """Horizontal bar chart of Random Forest feature importances."""
    fi = results["feature_importance"]
    features = list(fi.keys())
    values = list(fi.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features[::-1], values[::-1], color=BLUE)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Feature Importance — Random Forest Model", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/chart_feature_importance.png", dpi=150)
    plt.close()


def plot_cross_validation(results):
    """Bar chart with error bars showing 5-fold CV mean accuracy per model."""
    cv = results["cross_validation"]
    models = list(cv.keys())
    means = [cv[m]["mean_accuracy"] for m in models]
    stds = [cv[m]["std_accuracy"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, means, yerr=stds, capsize=8,
                   color=[BLUE, GREEN, RED], alpha=0.85)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=11)
    ax.set_title("5-Fold Cross-Validation Accuracy (mean ± std)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 3, f"{mean:.1f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/chart_cross_validation.png", dpi=150)
    plt.close()


def plot_confusion_matrices(results):
    """Confusion matrix heatmap for each of the three models, side by side."""
    models = list(results["model_results"].keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, model in zip(axes, models):
        cm = np.array(results["model_results"][model]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Predicted\nHealthy", "Predicted\nDiabetic"],
                    yticklabels=["Actual\nHealthy", "Actual\nDiabetic"],
                    annot_kws={"size": 14})
        ax.set_title(model, fontsize=11, fontweight="bold")

    fig.suptitle("Confusion Matrices — Test Set (n=154)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/chart_confusion_matrices.png", dpi=150)
    plt.close()


def plot_correlation_heatmap():
    """Correlation heatmap of all features plus the outcome variable."""
    df = pd.read_csv(DATA_PATH, header=None, names=[
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
        "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
    ])
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/chart_correlation_heatmap.png", dpi=150)
    plt.close()


def plot_class_distribution(results):
    """Bar chart showing the diabetic vs. non-diabetic class balance."""
    info = results["dataset_info"]
    labels = ["Non-diabetic", "Diabetic"]
    values = [info["non_diabetic_cases"], info["diabetic_cases"]]
    pct = [v / info["total_patients"] * 100 for v in values]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    bars = ax.bar(labels, values, color=[GREEN, RED], width=0.5)
    ax.set_ylabel("Number of patients", fontsize=11)
    ax.set_title(f"Class Distribution (n={info['total_patients']})",
                 fontsize=13, fontweight="bold")
    for bar, v, p in zip(bars, values, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 8, f"{v} ({p:.1f}%)",
                ha="center", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/chart_class_distribution.png", dpi=150)
    plt.close()


def main():
    results = load_results()
    plot_model_comparison(results)
    plot_roc_curves(results)
    plot_feature_importance(results)
    plot_cross_validation(results)
    plot_confusion_matrices(results)
    plot_correlation_heatmap()
    plot_class_distribution(results)
    print("All charts saved to the images/ directory.")


if __name__ == "__main__":
    main()
